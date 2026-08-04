

from assignment.part3.all import *

if __name__=="__main__":
    training_loop(# model
        vocab_size=10000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000,
        # data
        train_path="data/tiny_train.npy",
        valid_path='data/tiny_valid.npy',
        batch_size=32,
        # optimization
        total_steps=5000,
        lr_max=1e-3,
        lr_min=1e-4,
        warmup_iters=200,
        cosine_cycle_iters=5000,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_grad_norm=1.0,
        # infra
        device="mps",
        checkpoint_path="ckpt.pt",
        checkpoint_interval=1000,
        eval_interval=250,
        eval_iters=20,
        log_interval=50,
        seed=0,
        run_name="tinyv1")