# config settings file 

configDict = {
    'gpu_id': 7,
    'num_training_samples': 50,
    'is_binary': True,
    'batch_size': 128,
    'num_epochs': 1000,
    'optimizer': 'Adam',
    'scheduler': 'StepLR',
    'lr': 1e-3,
    'lr_step_size': 500,
    'lr_gamma': 0.9,
    'latent_dim': 256,
    'root_dir': 'data/isotropic',
    # save_dir : 'root_dir's child directory_num_training_samples_train_is_binary_BayesEnt_cvae_singleGPU_run'
    'save_dir': 'isotropic_true_50_trainSamples_binaryInput_CVAE',
    'load_model': False,
    'load_model_path': 'ConstrainedUQ/models/2024-01-10_20-50-14_isotropic_data_BayesEnt_cvae_DDP_run/checkpoints/checkpoint.pth',
    'description': 'ETA BEING OPTIMIZED FOR AS WEIGHT TO THE L1 LOSS.'
}