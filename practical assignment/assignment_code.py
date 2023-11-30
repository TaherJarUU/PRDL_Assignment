import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.linear_model._logistic import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure

mnist_data = pd.read_csv('mnist.csv').values
#print(mnist_data.value_counts(subset='label', normalize=True))
#print(mnist_data.describe())

labels = mnist_data[:, 0]
digits = mnist_data[:, 1:]

train_digits = digits[:37000]
test_digits = digits[37000:]

train_labels = labels[:37000]
test_labels = labels[37000:]
#img_size = 28
#plt.imshow(digits[0].reshape(img_size, img_size))
#plt.show()


'''By looking at the summary statistics we can see that the features (pixels) on the edges of the images in any direction do not provide 
any information about the content of the image and can be regarded as useless variables.'''

'''After examining the class distribution, we can see that, although the distribution is more or less equal for all classes, there is a 
significant difference between the majority class (digit number 1) and the least frequent class (digit number 5), a difference of 
more than %2 of the overall class counts. This means that if the classification was to default to the majority class all predictions 
would be digit 1, which would result in an accuracy of about %11'''

def index_distance():
    

    spread = []

    for idx, row in enumerate(digits):
        start = 0
        end = 28
        total = 784

        global_max_idx = 0
        global_min_idx = 0
        while end <= total:
            chunk = row[start:end]
            max_idx = np.argmax(chunk) + start
            min_idx = np.argmin(chunk) + start

            if max_idx > global_max_idx and chunk.any() != 0:
                global_max_idx = max_idx

            if global_min_idx == 0 and chunk.any() != 0:
                global_min_idx = min_idx
            start += 28
            end += 28
        spread.append(global_max_idx - global_min_idx)

    return spread


ink = np.array([sum(row) for row in train_digits])
#ink_mean = [np.mean(ink[labels == i]) for i in range(10)]
#ink_std = [np.std(ink[labels == i]) for i in range(10)]

ink = scale(ink).reshape(-1, 1)
classifier = LogisticRegression()
classifier.fit(ink,train_labels)

ink_test = np.array([sum(row) for row in test_digits])
#ink_mean = [np.mean(ink[labels == i]) for i in range(10)]
#ink_std = [np.std(ink[labels == i]) for i in range(10)]

ink_test = scale(ink_test).reshape(-1, 1)
predictions = classifier.predict(ink_test)
cm = confusion_matrix(test_labels, predictions)
#disp = ConfusionMatrixDisplay(confusion_matrix=cm,
#                             display_labels=classifier.classes_)
#disp.plot()
#plt.show()
class_report = classification_report(test_labels,predictions)
print(class_report)

#color_std = np.array([np.std(row) for row in digits])
#color_std = scale(color_std).reshape(-1, 1)

#img_length = np.array([np.sqrt(np.sum([i**2 for i in row])) for row in digits])
#img_length = scale(img_length).reshape(-1,1)

#index_spread = np.asarray(index_distance())
#index_spread = scale(index_spread).reshape(-1,1)

hog_feature = np.array([hog(np.reshape(row, (28, 28)), orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), feature_vector=True) for row in train_digits])

features = np.column_stack([ink,hog_feature])
features_df = pd.DataFrame(features)


classifier.fit(features,train_labels)

hog_feature_test = np.array([hog(np.reshape(row, (28, 28)), orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), feature_vector=True) for row in test_digits])
features_test = np.column_stack([ink_test,hog_feature_test])
predictions = classifier.predict(features_test)
cm = confusion_matrix(test_labels, predictions)

class_report = classification_report(test_labels,predictions)
print(class_report)
#disp = ConfusionMatrixDisplay(confusion_matrix=cm,
#                             display_labels=classifier.classes_)
#disp.plot()
#plt.show()

#image = digits[30]
#image = np.reshape(image, (28, 28))
#fd, hog_image = hog(image, orientations=8, pixels_per_cell=(2, 2), cells_per_block=(1, 1), visualize=True, feature_vector=True)


#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
#
#ax1.axis('off')
#ax1.imshow(image, cmap=plt.cm.gray)
#ax1.set_title('Input image')
#
## Rescale histogram for better display
#hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
#
#ax2.axis('off')
#ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
#ax2.set_title('Histogram of Oriented Gradients')
#plt.show()