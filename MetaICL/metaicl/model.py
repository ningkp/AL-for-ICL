
import numpy as np
import os
import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import Adafactor, AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForCausalLM

from ..utils.utils import get_checkpoint_id, download_file

class MetaICLModel(object):

    def __init__(self, logger=None, out_dir=None, fp16=True, local_rank=-1,args=None):
        if logger is None:
            class Logger():
                def info(self, text):
                    print ("Logging from MetaICLModel:\t", text)
            logger = Logger()

        self.logger = logger
        self.out_dir = out_dir
        self.fp16 = fp16
        self.local_rank = local_rank
        self.args = args

        if self.local_rank == -1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()
            ws = 1
        else:  # distributed mode
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            ws = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
            torch.distributed.init_process_group(backend="nccl")
            n_gpu = 1

        self.n_gpu = n_gpu
        self.device = device
        if self.local_rank <= 0:
            logger.info("Setting up for local_rank=%d, world_size=%d" % (self.local_rank, ws))
        self.model_name = None
        self.model = None
        self.mode = None

    def __str__(self):
        text = "[MetaICL Model]: "
        if self.model_name is None:
            text += "No model loaded yet"
        else:
            text += self.model_name
            if self.mode is None:
                text += " (no mode setted - try .train() or .eval()"
            else:
                text += " (%s mode)" % self.mode
        text += "\nusing device %s, %d gpus, local_rank=%d" % (self.device, self.n_gpu, self.local_rank)
        return ("="*50) + "\n" + text + "\n" + ("="*50)

    def is_none(self):
        return self.model is None

    def train(self):
        self.model.train()
        self.mode = "train"

    def eval(self):
        self.model.eval()
        self.mode = "eval"

    def cuda(self):
        self.model.cuda()

    def to_device(self):
        self.model.to(self.device)

    def load(self):
        prefix = ""
        model_name = self.args.model_name
        if "llama" in self.args.model_name:
            if '7B' in self.args.model_name:
                model_name = "models/llama-7b-hf"
            elif '13B' in self.args.model_name:
                model_name = "/home/ubuntu/llama_models/13B_hf"
            from transformers import LlamaForCausalLM
            model = LlamaForCausalLM.from_pretrained(prefix+model_name, device_map='auto')

        elif 'falcon' in self.args.model_name:
            if '7B' in self.args.model_name:
                model_name = "tiiuae/falcon-7b"
            elif '40B' in self.args.model_name:
                model_name = "tiiuae/falcon-40b"
            model = AutoModelForCausalLM.from_pretrained(prefix+model_name, trust_remote_code=True, device_map='auto')

        elif 'mosaic' in self.args.model_name:
            if '7B' in self.args.model_name:
                model_name = 'mosaicml/mpt-7b'
            model = AutoModelForCausalLM.from_pretrained(prefix+model_name, trust_remote_code=True, device_map='auto')

        elif 'gpt' in self.args.model_name:

            if self.args.model_name == 'gpt-neo-1.3B':
                model_name = "models/gpt-neo-1.3B"

            elif self.args.model_name == 'gpt-neo2':
                model_name = "EleutherAI/gpt-neo-2.7B"

            elif self.args.model_name == 'gpt-neox':
                model_name = "EleutherAI/gpt-neox-20b"

            else:
                model_name = 'models/gpt-j-6B'

            
            model = AutoModelForCausalLM.from_pretrained(prefix+model_name, device_map='auto')
        self.model_name = prefix+model_name
        self.model = model

    def save(self, step):
        if self.local_rank <= 0:
            model_state_dict = {key[7:] if key.startswith("module.") else key: value.cpu()
                                for key, value in self.model.state_dict().items()}
            torch.save(model_state_dict, os.path.join(self.out_dir, "model-{}.pt".format(step)))
            self.logger.info("Saving model parameters at step=%d" % step)

    def setup_optimizer(self, optimization, num_training_steps, lr, weight_decay, warmup_steps):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        if optimization=="adafactor":
            optimizer = Adafactor(optimizer_grouped_parameters,
                                  lr=lr,
                                  relative_step=False,
                                  warmup_init=False,
                                  weight_decay=weight_decay)
            scheduler = None
        elif optimization.startswith("adamw"):
            optimizer = AdamW(optimizer_grouped_parameters,
                              lr=lr,
                              eps=1e-08,
                              weight_decay=weight_decay)
            if self.fp16:
                self.model, optimizer = setup_fp16(self.model, optimizer)
            if optimization=="adamw":
                scheduler = get_linear_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=warmup_steps,
                                                            num_training_steps=num_training_steps)
            else:
                raise NotImplementedError()
        elif optimization=="8bit-adam":
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adam8bit(optimizer_grouped_parameters,
                                           lr=lr, betas=(0.9, 0.995))
            if self.fp16:
                self.model, optimizer = setup_fp16(self.model, optimizer)
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=warmup_steps,
                                                        num_training_steps=num_training_steps)
        else:
            raise NotImplementedError()

        self.optimizer = optimizer
        self.scheduler = scheduler

    def parallel(self):
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        if self.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank)


    def do_train(self, data, batch_size, num_training_steps, save_period, log_period,
                 gradient_accumulation_steps=1, max_grad_norm=1.0):
        dataloader = data.get_dataloader(batch_size, is_training=True)
        n_trainable_params = len([param for param in self.model.parameters() if param.requires_grad])
        n_gpus = torch.cuda.device_count()
        self.logger.info("Training {} parameters on {} examples for {} steps using {} GPUs".format(
            n_trainable_params, len(data), num_training_steps, self.n_gpu))

        global_step = 0
        train_losses = []
        best_accuracy = -1
        stop_training=False

        for epoch in range(num_training_steps):
            for batch in dataloader:
                global_step += 1

                input_ids=batch[0].to(self.device)
                attention_mask=batch[1].to(self.device)
                token_type_ids=batch[2].to(self.device)
                if len(batch)==3:
                    labels=None
                else:
                    labels=batch[3].to(self.device)

                loss = self.run_model(input_ids, attention_mask, token_type_ids, labels=labels)
                loss = loss.mean()

                if torch.isnan(loss).data:
                    print ("Stop training because loss=%s" % (loss.data))
                    stop_training=True
                    break
                train_losses.append(loss.detach().cpu())

                if self.fp16:
                    from apex import amp
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if global_step % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    self.optimizer.step()    # We have accumulated enought gradients
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.model.zero_grad()

                if global_step % log_period == 0:
                    self.logger.info("local rank %d\tglobal step %d\ttrain loss %.2f" % (self.local_rank, global_step, np.mean(train_losses)))
                    train_losses = []

                if global_step % save_period == 0:
                    self.save(global_step)

                if global_step==num_training_steps:
                    break

            if global_step==num_training_steps:
                break

        self.logger.info("Finish training")

    def do_inference(self, data, batch_size=1, verbose=False, calibration=False, do_probs=False):
        dataloader = data.get_dataloader(batch_size, is_training=False, calibration=calibration)
        if verbose:
            dataloader = tqdm(dataloader)
        losses = []

        for batch in dataloader:
            input_ids=batch[0].cuda()
            attention_mask=batch[1].cuda()
            token_type_ids=batch[2].cuda()
            if len(batch)==3:
                labels=None
            else:
                labels=batch[3].cuda()
            with torch.no_grad():
                loss = self.run_model(input_ids, attention_mask, token_type_ids, labels=labels)
            
            losses += loss.cpu().detach().numpy().tolist()
            
        if do_probs:
            losses = np.array(losses)
            def softmax(x):
                return(np.exp(x)/np.exp(x).sum())
            #print("Inside inference, losses", losses)
            probs = softmax(-losses)
            return probs
        # print(probs)
        else:
            return losses

    def do_predict(self, data, batch_size=1, losses=None, verbose=False, require_loss=False, label_id=None, calibration=False, do_probs=False):
        
        losses = self.do_inference(data, batch_size, verbose=verbose, do_probs=do_probs)
        losses = np.array(losses)
        assert len(losses) == len(data)
        predictions = []
        for idx, dp in enumerate(data.metadata):
            
            curr_label_losses = [np.sum(losses[indices]) for indices in dp["indices"]]
            
            if label_id is not None:
                prediction = dp["options"][label_id]
                negative_pred_prob = curr_label_losses[label_id]
                predictions.append([prediction.strip(), negative_pred_prob])
            if not require_loss:
                prediction_idx = sorted(enumerate(curr_label_losses), key=lambda x: x[1])[0][0]
                prediction = dp["options"][prediction_idx]
                predictions.append(prediction.strip())
            else:
                if do_probs:
                    reverse=True
                else:
                    reverse=False
                prediction_terms = sorted(enumerate(curr_label_losses), key=lambda x: x[1], reverse=reverse)[0]

                prediction = dp["options"][prediction_terms[0]]
                negative_pred_prob = prediction_terms[1]
                predictions.append([prediction.strip(), negative_pred_prob])

        return predictions
    
    

    def run_model(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[..., :-1, :].contiguous()

        if labels is None:
            labels = input_ids
        labels = labels[..., 1:].contiguous()
       

        #label mask filters out non-labels
        label_mask = token_type_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)) # [batch_size, length]

        losses = losses.view(logits.size(0), logits.size(1)) * label_mask
        norm_losses = torch.sum(losses, axis=1) / torch.sum(label_mask, axis=1)
        
        return norm_losses 
    

def setup_fp16(model, optimizer):
    try:
        import apex
        from apex import amp
        apex.amp.register_half_function(torch, "einsum")
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    fp16_opt_level = "O1"
    model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)
    return model, optimizer



