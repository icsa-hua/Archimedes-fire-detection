from iqa.fiqe import train_fiqe


train_fiqe(inputVideoPath='./data/1.MP4', frame_step=10, patch_size=96, stride=96, output_path='fiqe_model.npz')