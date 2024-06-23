# Grid Search Template
A python template to run grid search for some hyperparameters

Uses idle gpus and can run trains in parallel (currently supports only single gpu training). The run command and hyperparameters should be modified based on the requirements.

The run command in the script:

    subprocess.Popen(combined_pretrain_and_finetune_command, shell=True)
