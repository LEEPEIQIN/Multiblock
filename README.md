# SR_multiblock


The porcedure follows the phased training step:

Phase1-1: using LR and HR images to train N_1 (python phase1-1.py)

Phase1-2: using pretrained N_1 and LR to general a new low-resolution data set LR_1 (python phase1-2.py )

Phase 2-1: using LR_1 and HR images to train N_2 (python phase1-1.py)

Phase 2-2: using pretrained N_2 and LR_1 to general a new low-resolution data set LR_2 (python phase1-2.py)

Phase 3: using LR_2 and HR images to train N_3(python phase3-1.py)

Phase 4 (combined training): initialize the total net work N with pretrained network block 
N_1,N_2,N_3 and use backpropagation to further modify the parameters.(python phase4.py)


Please note that for phase1-2,2-2,3-2 you may need to delete the existing image folder ( 2_LR_afternet1; 2_LR_afternet2 2_LR_afternet3) first in order to generate new image.
