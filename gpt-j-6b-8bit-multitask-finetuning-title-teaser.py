# To get error messages
import traceback

# Mail imports
import smtplib
from email.mime.text import MIMEText

# FTP client to transfer data
import pysftp


import time
import os

import transformers

# Torch Imports
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import custom_fwd, custom_bwd
from torch.utils.data import DataLoader


from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
from bitsandbytes.optim import Adam8bit

from datasets import load_dataset

# Configure settings for ftp model checkpoint transfer.
FTP_IP = "Add.Your.IP.Adress.Here"
FTP_USER = "your-username"
FTP_PASSWORD = "your-password"


# Configure mail credentials to get progress notifications via e-mail.
MAIL_SENDER = "Add your email here"
MAIL_RECEIVERS = [MAIL_SENDER]
MAIL_SERVER = "add.your.mailserver.here"
MAIL_USER = "your-user"
MAIL_PASSWORD = "your-password"
SEND_MAIL_NOTIFICATIONS = False

# Given a message send mail
def send_mail(message):
    sender = MAIL_SENDER
    receivers = MAIL_RECEIVERS


    port = 25
    msg = MIMEText(message)

    msg['Subject'] = 'Learning Progress'
    msg['From'] = MAIL_SENDER
    msg['To'] = MAIL_RECEIVERS

    with smtplib.SMTP(MAIL_SERVER, port) as server:
        server.login(MAIL_USER, MAIL_PASSWORD)
        server.sendmail(sender, receivers, msg.as_string())
        print("Successfully sent email")


if SEND_MAIL_NOTIFICATIONS:
    send_mail("Start Script.")

# Wrap script in a main() function
def main():

    """## Convert Model to 8 bits
    We convert EleutherAI's GPT-J-6B model to 8 bits using facebook's [bitsandbytes](https://github.com/facebookresearch/bitsandbytes) library. This reduces the model's size from 20Gb down to 6Gb.
    * large weight tensors are quantized using dynamic 8-bit quantization and de-quantized just-in-time for multiplication
    * using gradient checkpoints to store one only activation per layer: using dramatically less memory at the cost of 30% slower training
    """

    class FrozenBNBLinear(nn.Module):
        def __init__(self, weight, absmax, code, bias=None):
            assert isinstance(bias, nn.Parameter) or bias is None
            super().__init__()
            self.out_features, self.in_features = weight.shape
            self.register_buffer("weight", weight.requires_grad_(False))
            self.register_buffer("absmax", absmax.requires_grad_(False))
            self.register_buffer("code", code.requires_grad_(False))
            self.adapter = None
            self.bias = bias
    
        def forward(self, input):
            output = DequantizeAndLinear.apply(input, self.weight, self.absmax, self.code, self.bias)
            if self.adapter:
                output += self.adapter(input)
            return output
    
        @classmethod
        def from_linear(cls, linear: nn.Linear) -> "FrozenBNBLinear":
            weights_int8, state = quantize_blockise_lowmemory(linear.weight)
            return cls(weights_int8, *state, linear.bias)
    
        def __repr__(self):
            return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"
    
    
    class DequantizeAndLinear(torch.autograd.Function): 
        @staticmethod
        @custom_fwd
        def forward(ctx, input: torch.Tensor, weights_quantized: torch.ByteTensor,
                    absmax: torch.FloatTensor, code: torch.FloatTensor, bias: torch.FloatTensor):
            weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
            ctx.save_for_backward(input, weights_quantized, absmax, code)
            ctx._has_bias = bias is not None
            return F.linear(input, weights_deq, bias).clone()
    
        @staticmethod
        @custom_bwd
        def backward(ctx, grad_output: torch.Tensor):
            assert not ctx.needs_input_grad[1] and not ctx.needs_input_grad[2] and not ctx.needs_input_grad[3]
            input, weights_quantized, absmax, code = ctx.saved_tensors
            # grad_output: [*batch, out_features]
            weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
            grad_input = grad_output @ weights_deq
            grad_bias = grad_output.flatten(0, -2).sum(dim=0) if ctx._has_bias else None
            return grad_input, None, None, None, grad_bias
    
    
    class FrozenBNBEmbedding(nn.Module):
        def __init__(self, weight, absmax, code):
            super().__init__()
            self.num_embeddings, self.embedding_dim = weight.shape
            self.register_buffer("weight", weight.requires_grad_(False))
            self.register_buffer("absmax", absmax.requires_grad_(False))
            self.register_buffer("code", code.requires_grad_(False))
            self.adapter = None
    
        def forward(self, input, **kwargs):
            with torch.no_grad():
                # note: both quantized weights and input indices are not differentiable
                weight_deq = dequantize_blockwise(self.weight, absmax=self.absmax, code=self.code)
                output = F.embedding(input, weight_deq, **kwargs)
            if self.adapter:
                output += self.adapter(input)
            return output 
    
        @classmethod
        def from_embedding(cls, embedding: nn.Embedding) -> "FrozenBNBEmbedding":
            weights_int8, state = quantize_blockise_lowmemory(embedding.weight)
            return cls(weights_int8, *state)
    
        def __repr__(self):
            return f"{self.__class__.__name__}({self.num_embeddings}, {self.embedding_dim})"
    
    
    def quantize_blockise_lowmemory(matrix: torch.Tensor, chunk_size: int = 2 ** 20):
        assert chunk_size % 4096 == 0
        code = None
        chunks = []
        absmaxes = []
        flat_tensor = matrix.view(-1)
        for i in range((matrix.numel() - 1) // chunk_size + 1):
            input_chunk = flat_tensor[i * chunk_size: (i + 1) * chunk_size].clone()
            quantized_chunk, (absmax_chunk, code) = quantize_blockwise(input_chunk, code=code)
            chunks.append(quantized_chunk)
            absmaxes.append(absmax_chunk)
    
        matrix_i8 = torch.cat(chunks).reshape_as(matrix)
        absmax = torch.cat(absmaxes)
        return matrix_i8, (absmax, code)
    
    
    def convert_to_int8(model):
        """Convert linear and embedding modules to 8-bit with optional adapters"""
        for module in list(model.modules()):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    print(name, child)
                    setattr(
                        module,
                        name,
                        FrozenBNBLinear(
                            weight=torch.zeros(child.out_features, child.in_features, dtype=torch.uint8),
                            absmax=torch.zeros((child.weight.numel() - 1) // 4096 + 1),
                            code=torch.zeros(256),
                            bias=child.bias,
                        ),
                    )
                elif isinstance(child, nn.Embedding):
                    setattr(
                        module,
                        name,
                        FrozenBNBEmbedding(
                            weight=torch.zeros(child.num_embeddings, child.embedding_dim, dtype=torch.uint8),
                            absmax=torch.zeros((child.weight.numel() - 1) // 4096 + 1),
                            code=torch.zeros(256),
                        )
                    )

    class GPTJBlock(transformers.models.gptj.modeling_gptj.GPTJBlock):
        def __init__(self, config):
            super().__init__(config)

            convert_to_int8(self.attn)
            convert_to_int8(self.mlp)


    class GPTJModel(transformers.models.gptj.modeling_gptj.GPTJModel):
        def __init__(self, config):
            super().__init__(config)
            convert_to_int8(self)
            

    class GPTJForCausalLM(transformers.models.gptj.modeling_gptj.GPTJForCausalLM):
        def __init__(self, config):
            super().__init__(config)
            convert_to_int8(self)


    transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock

    """We use the model configuration and the tokenizer of GPT-J-6B and set `pad_token` as `eos_token`.

    """

    config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
    config.pad_token_id = config.eos_token_id

    tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer.pad_token = config.pad_token_id

    """## Load pretrained model
    We load the pre-trained gpt-j-6b with 8-bit weights from [huggingface](https://huggingface.co/hivemind/gpt-j-6B-8bit). To reduce the peak RAM usage, we add the argument, `low_cpu_mem_usage=True` to `from_pretrained`.
    """
    print('loaded tokenizer')
    gpt = GPTJForCausalLM.from_pretrained("hivemind/gpt-j-6B-8bit", low_cpu_mem_usage=True)
    print('loaded gptj')
    """## Add LoRA Adapter
    Low-Rank Adaptation, or LoRA, which freezes the pretrained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks.
    * We set `adapter_dim` from 16 to 4
    * We set the Dropout `p` from 0 to 0.1
    """

    def add_adapters(model, adapter_dim=4, p = 0.1):
        assert adapter_dim > 0

        for name, module in model.named_modules():
            if isinstance(module, FrozenBNBLinear):
                if "attn" in name or "mlp" in name or "head" in name:
                    print("Adding adapter to", name)
                    module.adapter = nn.Sequential(
                        nn.Linear(module.in_features, adapter_dim, bias=False),
                        nn.Dropout(p=p),
                        nn.Linear(adapter_dim, module.out_features, bias=False),
                    )
                    print("Initializing", name)
                    nn.init.zeros_(module.adapter[2].weight)

                else:
                    print("Not adding adapter to", name)
            elif isinstance(module, FrozenBNBEmbedding):
                print("Adding adapter to", name)
                module.adapter = nn.Sequential(
                        nn.Embedding(module.num_embeddings, adapter_dim),
                        nn.Dropout(p=p),
                        nn.Linear(adapter_dim, module.embedding_dim, bias=False),
                    )
                print("Initializing", name)
                nn.init.zeros_(module.adapter[2].weight)

    add_adapters(gpt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpt.to(device)

    """## Load Dataset
    We load our dataset as csv files from Google Drive. Data Structure:

    | pair                                              |   |   |   |   |
    |---------------------------------------------------|---|---|---|---|
    | [Text:] Lorem Ipsum<br>[Title:] Title of Lorem    |   |   |   |   |
    | [Text:] Dolor Sit Amet<br>[Title:] Title of Dolor |   |   |   |   |
    |                                                   |   |   |   |   |
    """


    print('added adapters')
    dataset = load_dataset("json", data_files='20221216dbk10k_MultiTaskCut.json')
    print('loaded dataset')
    """## Tokenize Data
    We tokenize our dataset and set `max_length=2048` for our task. Since GPTJ models operate with a maximum total token count of 2048 tokens, one Text-Title-Pair must not exceed this token count. You can use [this tool](https://beta.openai.com/tokenizer) to understand how a piece of text would be tokenized and the total count of tokens in that piece of text.
    """

    def tokenize_function(examples):
        return tokenizer(examples["pair"], padding=True, truncation=True, max_length=2048)
    

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["pair"])

    print('tokenized data')

    #Convert to Torch Format
    tokenized_datasets.set_format("torch")

    full_train_dataset = tokenized_datasets["train"]
    train_dataloader = DataLoader(full_train_dataset, shuffle=False, batch_size=8)




    """## Training Params
    We define the training parameters and initialize the Optimizer, Schedluer and Scaler.
    """

    num_epochs = 5
    num_training_steps = num_epochs * len(train_dataloader)
    num_warmup_steps = int(num_training_steps*0.1)
    learning_rate = 1e-5 # Initial Learning Rate of 0.00001

    """We activate gradient checkpointing for the model to reduce memory load."""

    gpt.gradient_checkpointing_enable()

    """We set a savepath for the model"""

    filepath = 'modelMultitaks20221219.pt'

    """We initialize the Optimizer, Schedluer and Scaler for training"""

    optimizer = Adam8bit(gpt.parameters(), lr=learning_rate, weight_decay=0.01)
    lr_scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    scaler = torch.cuda.amp.GradScaler()

    """## Training Loop
    We train the model and save it 👏
    """

    

    start = time.time()
    #progress_bar = tqdm(range(num_training_steps))
    gpt.train()
    k = 0

    if SEND_MAIL_NOTIFICATIONS:
        send_mail("Start Learning.")

    # Iterate through epochs
    for epoch in range(num_epochs):
        if SEND_MAIL_NOTIFICATIONS:
            send_mail(f"Start epoch {epoch}")
        # Iterate through our training data in batches
        for batch in train_dataloader:
            k = k + 1
            if k % 500 == 0:
                print(k)

                # Save Model with Torch
                #torch.save(gpt, filepath)
                torch.save(gpt.state_dict(), filepath)

                if SEND_MAIL_NOTIFICATIONS:
                    send_mail('Saved')

            # Define batch to train on
            batch = {k: v.to(device) for k, v in batch.items()}

            # Zeroing out the gradients
            # Explanation: https://stackoverflow.com/a/48009142
            optimizer.zero_grad()

            # Runs the forward pass with autocasting
            with torch.cuda.amp.autocast():

                # Feed batch in custom forward function
                out = gpt.forward(**batch,)
                
                # Custom loss function
                loss = F.cross_entropy(out.logits[:, :-1, :].flatten(0, -2),
                                    batch['input_ids'][:, 1:].flatten(),
                                    reduction='mean', 
                                    label_smoothing=0.1)

            print('k', k * 8, 'time', time.time() - start,'epoch', epoch, loss.item())

            # Scales loss
            # Calls backward() on scaled loss to create scaled gradients
            scaler.scale(loss).backward()

            # Unscales gradients held by optimizer’s assigned parameters
            scaler.unscale_(optimizer)

            # clips the norm of the overall gradient by concatenating all parameters passed to the function
            # documentation: https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html
            torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)

            # Updates the scale for next iteration.
            scaler.step(optimizer)
            scaler.update()

            # Updates initial learning rate
            lr_scheduler.step()

            #progress_bar.update(1)
            # break

    if SEND_MAIL_NOTIFICATIONS:
        send_mail('Trained')

    # Save Model with Torch
    torch.save(gpt.state_dict(), filepath)

    if SEND_MAIL_NOTIFICATIONS:
        send_mail('Saved')

    # Transfer model
    cnopts = pysftp.CnOpts()
    cnopts.hostkeys = None   
    with pysftp.Connection(FTP_IP, username=FTP_USER, password=FTP_PASSWORD, cnopts=cnopts) as sftp:
        with sftp.cd('/home/ai'):
            sftp.put(filepath)

    if SEND_MAIL_NOTIFICATIONS:
        send_mail('Transfered')
    
    print('destroy')
    
    # Destroy the instance
    os.system('python vast.py destroy instance 5654171')
    print('destroyed')

    if SEND_MAIL_NOTIFICATIONS:
        send_mail('Destroyed')


if __name__ == '__main__':
    # Wrap main() function in a try-except. So if an error occurs, we can send the message as a mail
    try:
        main()
    except:
        print(traceback.format_exc())
        with open('error.txt', 'w') as f:
            f.write(traceback.format_exc())
        if SEND_MAIL_NOTIFICATIONS:
            send_mail(traceback.format_exc())
