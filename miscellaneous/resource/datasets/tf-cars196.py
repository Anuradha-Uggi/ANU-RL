'''
import tensorflow_datasets as tfds

# Load the Cars196 dataset
dataset_train = tfds.load('cars196', split='train', as_supervised=True)
dataset_test = tfds.load('cars196', split='test', as_supervised=True)

# Iterate through the dataset
for image, label in dataset_train:
    # Your processing code here
    print('size check:', image.shape); print(hello)'''
    
import deeplake
ds_train = deeplake.load("hub://activeloop/stanford-cars-train")
ds_test = deeplake.load("hub://activeloop/stanford-cars-test")    

print('ds_train:', len(ds_train), len(ds_test))

# Check available keys (to see the structure)
print(ds_train.tensors)

# Access images and labels
image_data = ds_train.images[:]
labels = ds_train.boxes[:]

print('images shape checking...\n', len(image_data), len(labels))
for ind in labels:
    print('labels checking...:',ind)
print(hello)

# Display an example image with its label
index = 0  # Change this to see different images
image = np.array(image_data[index])  # Convert to NumPy array
label = labels[index]

# Plot the image
plt.imshow(image)
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()
