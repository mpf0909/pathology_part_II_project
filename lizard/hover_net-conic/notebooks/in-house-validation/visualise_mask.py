import numpy as np
import matplotlib.pyplot as plt

labels = np.load('/rds/user/mf774/hpc-work/part_II_project/in-house/training-CD3/data/labels.npy', mmap_mode='r')
images = np.load('/rds/user/mf774/hpc-work/part_II_project/in-house/training-CD3/data/images.npy', mmap_mode='r')
true_npy = np.load('/rds/user/mf774/hpc-work/part_II_project/in-house/training-CD3/validation/fold_1/valid_true.npy', mmap_mode='r')
pred_npy = np.load('/rds/user/mf774/hpc-work/part_II_project/in-house/training-CD3/validation/fold_1/valid_pred.npy', mmap_mode='r')

index = 0

patch_of_interest = labels[10002,:,:,1]
image_of_interest = images[10002,:,:,:]
true_npy_of_interest = true_npy[500,:,:,1]
pred_npy_of_interest = pred_npy[500,:,:,1]

print(len(labels))
print(len(images))
print(len(true_npy))
print(len(pred_npy))

plt.imshow(patch_of_interest)
plt.savefig('test_mask.png')
plt.imshow(image_of_interest)
plt.savefig('test_image.png')
plt.imshow(true_npy_of_interest)
plt.savefig('true_npy_of_interest.png')
plt.imshow(pred_npy_of_interest)
plt.savefig('pred_npy_of_interest.png')