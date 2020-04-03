import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader


from ..model import UNILMLM, UNILM
from .optim_schedule import ScheduledOptim
from ..util import logger, init_logger 

import tqdm


class UNILMTrainer:
    """
    UNILMTrainer make the pretrained UNILM model with 4 differnet mask schemes
        1. Bidirectioanl LM
        2. Left-to-Right LM
        3. Right-to-Left LM
        4. Seq-to-Seq LM
    
    Each mask is applied with equal probability during training

    """

    def __init__(self, unilm: UNILM, vocab: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, log_dir=None, args=None):
        """
        :param unilm: UNILM model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        :param log_dir: path for a directory for which log file is saved
        """
        self.vocab = vocab
        vocab_size = len(self.vocab)
        
        # Initialize Log file
        self.log_dir = log_dir
        init_logger(self.log_dir+'/log_{}.txt'.format(args.epochs))
        
        # Setup cuda device for UNILM training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        print('device : {}'.format(self.device))
        
        # This UNILM model will be saved every epoch
        self.unilm = unilm
        # Initialize the UNILM Language Model, with UNILM model
        self.model = UNILMLM(unilm, vocab_size).to(self.device)
        
        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for UNILM" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.unilm.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0) # vocab.pad_idx == 1

        self.log_freq = log_freq
        self.args = args
        
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.train_iteration(epoch, self.train_data)

    def train_iteration(self, epoch, data_loader):
        """
        loop over the data_loader for training
        and also auto save the model at the end of train
        
        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :return: None
        """
        self.model.train()
        total_loss = 0.0
        global_step = 0
        best_eval_acc = 0.0
        best_eval_loss = 18.0
        
        for e in range(epoch):
            for i, data in enumerate(data_loader):
                # 0. batch_data will be sent into the device(GPU or cpu)
                data = {key: value.to(self.device) for key, value in data.items()}

                # 1. forward the next_sentence_prediction and masked_lm model
                mask_lm_output = self.model.forward(data["unilm_input"], data["segment_label"], data["unilm_mask"])

                # 2. NLLLoss of predicting masked token word
                loss = self.criterion(mask_lm_output.transpose(1, 2), data["unilm_label"])

                # 3. backward and optimization only in train
           
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()
  
                total_loss += loss.item()
                
                # Masked token prediction accuracy
#                 print(mask_lm_output.max(dim=-1)[1])
#                 print('')
#                 print(data["unilm_label"])
#                 print('******************')
                acc = (mask_lm_output.max(dim=-1)[1] == data["unilm_label"]).float()[data["unilm_label"] != self.vocab.unk_idx].mean()
            
                if i % self.log_freq == 0:
                    global_step+= self.log_freq
                    logger.info('[Train]  <epoch>  {}/{}  <step>  {}  <avg_loss>  {:.3f} <loss>  {:.3f}  <acc>  {:.3%}'.format(e+1, epoch, global_step, total_loss/global_step, loss.item(), acc.item()))

                    # Evaluation
                    if self.args.do_test and global_step % self.args.test_every==0:

                        eval_loss, eval_acc = self.test(self.test_data)
                        logger.info('[Evaluation] <epoch>  {}/{}  <step>  {}  <eval_loss>  {:.3f}  <eval_acc>  {:.3%}'.format(e+1, epoch, global_step, eval_loss, eval_acc))
            
                        if self.args.save_during_train and  eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            save_path = self.log_dir+'/model_{}_eval_loss_{:.3f}.pt'.format(e, best_eval_loss)
                            self.save(e, save_path)

        
        logger.info('********* Train finished *********')
        
        # Model Save
        save_path = self.log_dir+'/model_{}_eval_loss_{:.3f}.pt'.format(e, eval_loss)
        self.save(e, save_path)
        
 
    def test(self, data_loader):
        """
        loop over the data_loader for testing

        :param data_loader: torch.utils.data.DataLoader for iteration
        :return: eval_loss
        """

        self.model.eval()
        for i, data in enumerate(data_loader):
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            mask_lm_output = self.model.forward(data["unilm_input"], data["segment_label"], data["unilm_mask"])

            # 2. NLLLoss of predicting masked token word
            loss = self.criterion(mask_lm_output.transpose(1, 2), data["unilm_label"])
           
            # 3. Masked token prediction accuracy
            acc = (mask_lm_output.max(dim=-1)[1] == data["unilm_label"]).float()[data["unilm_label"] != self.vocab.pad_idx].mean()
            
        
        return loss.item(), acc.item()

        

    def save(self, epoch, save_path):
        """
        Saving the current UNILM model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        self.unilm.cpu()
        state = {'epoch': epoch,
                 'maxlen': self.args.seq_len,
                 'layer': self.args.layers,
                 'hidden': self.args.hidden,
                 'model_state_dict': self.unilm.state_dict(),
                 'opt_state_dict':self.optim.state_dict()
                }
        torch.save(state, save_path)
        logger.info('Model saved to {}'.format(save_path))
        self.unilm.to(self.device)
