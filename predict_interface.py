# import tensorflow as tf
import os
import sys
from scipy import ndimage
import numpy as np
import cv2
import input_data
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class PredSet(object):
    def __init__(self, location, top_left=None, bottom_right=None, actual_w_h=None, prob_with_pred=None):
        self.location = location

        if top_left is None:
            top_left = []
        else:
            self.top_left = top_left

        if bottom_right is None:
            bottom_right = []
        else:
            self.bottom_right = bottom_right

        if actual_w_h is None:
            actual_w_h = []
        else:
            self.actual_w_h = actual_w_h

        if prob_with_pred is None:
            prob_with_pred = []
        else:
            self.prob_with_pred = prob_with_pred

    def get_location(self):
        return self.location

    def get_top_left(self):
        return self.top_left

    def get_bottom_right(self):
        return self.bottom_right

    def get_top(self):
        return self.top_left[0]

    def get_left(self):
        return self.top_left[1]

    def get_bottom(self):
        return self.bottom_right[0]

    def get_right(self):
        return self.bottom_right[1]

    def get_actual_w_h(self):
        return self.actual_w_h

    def get_actual_width(self):
        return self.actual_w_h[0]

    def get_actual_height(self):
        return self.actual_w_h[1]

    def get_prediction(self):
        return self.prob_with_pred[1]

    def get_probability(self):
        return self.prob_with_pred[0]

    def print_to_output(self):
        print("Prediction for ", self.location)
        print("Pos")
        print(self.top_left)
        print(self.bottom_right)
        print(self.actual_w_h)
        print(" ")
        print(self.prob_with_pred)


def getBestShift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)
    print(cy, cx)

    rows, cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx, shifty


def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


class MicroObject(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Predictor(object):
    def __init__(self):
        #################################
        # Initialize the neural network #
        #################################
        """
        A placeholder for our image data:
        None stands for an unspecified number of images
        """
        input_size = 28*28  # 784 = 784 * 784 pixels
        x = tf.placeholder("float", [None, input_size])

        # We need our weights for our neural net...
        W = tf.Variable(tf.zeros([input_size, 10]))

        # ...and the biases
        b = tf.Variable(tf.zeros([10]))

        """
        Softmax provides a probability based output.
        We need to multiply the image values x and the weights
        and add the biases
        (the normal procedure, explained in previous articles)
        """
        y = tf.nn.softmax(tf.matmul(x, W) + b)

        """
        y_ will be filled with the real values
        which we want to train (digits 0-9)
        for an undefined number of images
        """
        y_ = tf.placeholder("float", [None, 10])

        """
        we use the cross_entropy function
        which we want to minimize to improve our model
        """
        cross_entropy = -tf.reduce_sum(y_*tf.log(y))

        """
        use a learning rate of 0.01
        to minimize the cross_entropy error
        """
        train_step = tf.train.GradientDescentOptimizer(
            0.01).minimize(cross_entropy)

        self.model = MicroObject(
            x=x,
            W=W,
            b=b,
            y=y,
            y_=y_,
            cross_entropy=cross_entropy,
            train_step=train_step
        )

        ##########################################
        # Initialize the training infrastructure #
        ##########################################

        self.dataset_dir = "MNIST_data/"
        self.checkpoint_dir = "cps/"
        self.checkpoint_file = self.checkpoint_dir+'model.ckpt'

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        # Initialize all variables and run init
        self.sess.run(tf.initialize_all_variables())

    def train(self):
        # Create a MNIST_data folder with the MNIST dataset if necessary
        mnist = input_data.read_data_sets(self.dataset_dir, one_hot=True)

        # Use 1000 batches with a size of 100 each to train our net
        for i in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            # Run the train_step function with the given image values (x) and the real output (y_)
            self.sess.run(self.model.train_step, feed_dict={
                          self.model.x: batch_xs, self.model.y_: batch_ys})

        self.saver.save(self.sess, self.checkpoint_file)

        """
		Let's get the accuracy of our model:
		our model is correct if the index with the highest y value
		is the same as in the real digit vector
		The mean of the correct_prediction gives us the accuracy.
		We need to run the accuracy function
		with our test set (mnist.test)
		We use the keys "images" and "labels" for x and y_
		"""
        correct_prediction = tf.equal(
            tf.argmax(self.model.y, 1), tf.argmax(self.model.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(self.sess.run(accuracy, feed_dict={
              self.model.x: mnist.test.images, self.model.y_: mnist.test.labels}))

    def load_checkpoint(self):
        # Here's where you're restoring the variables w and b.
        # Note that the graph is exactly as it was when the variables were
        # saved in a prior training run.
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint found')
            exit(1)

        mnist = input_data.read_data_sets(self.dataset_dir, one_hot=True)
        correct_prediction = tf.equal(
            tf.argmax(self.model.y, 1), tf.argmax(self.model.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("accuracy: ", sess.run(accuracy, feed_dict={
              self.model.x: mnist.test.images, self.model.y_: mnist.test.labels}))

    def get_prediction_and_probability(self, flatten):
        prediction = [tf.reduce_max(self.model.y),
                      tf.argmax(self.model.y, 1)[0]]
        prediction_and_probability = self.sess.run(
            prediction, feed_dict={self.model.x: [flatten]})
        return prediction_and_probability


class PredictionSession(object):
    def __init__(self, predictor):
        self.predictor = predictor

    def load_image(self, image):
        # image_path = "img/" + image + ".png"
        self.image_path = image
        self.image_name = os.path.basename(image)

        if not os.path.exists(self.image_path):
            print("File " + self.image_path + " doesn't exist")
            exit(1)

        self.output_base_path = "pro-img/" + self.image_name + "/"

        if not os.path.exists(self.output_base_path):
            os.makedirs(self.output_base_path)

        # Read original image with colors
        self.color_complete = cv2.imread(self.image_path)

        # Read original image as grayscale
        self.gray_complete = cv2.imread(self.image_path, 0)

        # Create a black & white version of the grayscale image
        # (which we will perform the detection on)
        _, self.gray_complete = cv2.threshold(
            255-self.gray_complete, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Debugging: Write the black & white image
        cv2.imwrite(self.output_base_path + "compl.png", self.gray_complete)

        self.digit_image = -np.ones(self.gray_complete.shape)

        self.height, self.width = self.gray_complete.shape

    def crop_images(self):
        pass

    def run_prediction(self):
        self.predictions = []

        """
        crop into several images
        """
        for cropped_width in range(100, 300, 20):
            for cropped_height in range(100, 300, 20):
                for shift_x in range(0, self.width-cropped_width, int(cropped_width/4)):
                    for shift_y in range(0, self.height-cropped_height, int(cropped_height/4)):
                        gray = self.gray_complete[shift_y:shift_y +
                                                  cropped_height, shift_x:shift_x + cropped_width]
                        if np.count_nonzero(gray) <= 20:
                            continue

                        if (np.sum(gray[0]) != 0) or (np.sum(gray[:, 0]) != 0) or (np.sum(gray[-1]) != 0) or (np.sum(gray[:,
                                                                                                                          -1]) != 0):
                            continue

                        top_left = np.array([shift_y, shift_x])
                        bottom_right = np.array(
                            [shift_y+cropped_height, shift_x + cropped_width])

                        while np.sum(gray[0]) == 0:
                            top_left[0] += 1
                            gray = gray[1:]

                        while np.sum(gray[:, 0]) == 0:
                            top_left[1] += 1
                            gray = np.delete(gray, 0, 1)

                        while np.sum(gray[-1]) == 0:
                            bottom_right[0] -= 1
                            gray = gray[:-1]

                        while np.sum(gray[:, -1]) == 0:
                            bottom_right[1] -= 1
                            gray = np.delete(gray, -1, 1)

                        actual_w_h = bottom_right-top_left
                        if (np.count_nonzero(self.digit_image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]+1) >
                                0.2*actual_w_h[0]*actual_w_h[1]):
                            continue

                        print("------------------")
                        print("------------------")

                        rows, cols = gray.shape
                        compl_dif = abs(rows-cols)
                        half_Sm = int(compl_dif/2)
                        half_Big = half_Sm if half_Sm*2 == compl_dif else half_Sm+1
                        if rows > cols:
                            gray = np.lib.pad(
                                gray, ((0, 0), (half_Sm, half_Big)), 'constant')
                        else:
                            gray = np.lib.pad(
                                gray, ((half_Sm, half_Big), (0, 0)), 'constant')

                        gray = cv2.resize(gray, (20, 20))
                        gray = np.lib.pad(gray, ((4, 4), (4, 4)), 'constant')

                        shiftx, shifty = getBestShift(gray)
                        shifted = shift(gray, shiftx, shifty)
                        gray = shifted

                        cv2.imwrite(self.output_base_path+"shifted"+"_" +
                                    str(shift_x)+"_"+str(shift_y)+".png", gray)

                        """
                        All images in the training set have a range from 0-1
                        and not from 0-255, so we divide our flattened images
                        (a one-dimensional vector with our 784 pixels)
                        by 255 to use the same 0-1 based range.
                        """
                        flatten = gray.flatten() / 255.0

                        pred = self.predictor.get_prediction_and_probability(
                            flatten)

                        prediction_result = PredSet((shift_x, shift_y, cropped_width),
                                                    top_left, bottom_right, actual_w_h, pred)

                        prediction_result.print_to_output()
                        self.predictions.append(prediction_result)

                        self.store_digit_image(prediction_result)
                        self.annotate_image(prediction_result)

    def store_digit_image(self, prediction_result):
        p = prediction_result

        self.digit_image[p.get_top():p.get_bottom(),
                         p.get_left():p.get_right()] = p.get_prediction()

    def annotate_image(self, prediction_result):
        p = prediction_result

        # Draw a rectangle around the area
        cv2.rectangle(self.color_complete,
                      (p.get_left(), p.get_top()),
                      (p.get_right(), p.get_bottom()),
                      color=(0, 255, 0), thickness=5)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Write the prediction value {0,1,..,9} next to the rectangle
        cv2.putText(self.color_complete, str(p.get_prediction()),
                    (p.get_left(), p.get_bottom()+50),
                    font, fontScale=1.4, color=(0, 255, 0), thickness=4)

        # Write the probability [0%, 100%] next to it
        cv2.putText(self.color_complete, format(p.get_probability()*100, ".1f")+"%",
                    (p.get_left()+30, p.get_bottom()+60),
                    font, fontScale=0.8, color=(0, 255, 0), thickness=2)

    def write_annotated_image(self):
        cv2.imwrite(self.output_base_path +
                    "digitized_image.png", self.color_complete)


def pred_from_img(image, train):
    image = image
    train = train

    predictor = Predictor()
    pred_sess = PredictionSession(predictor)

    pred_sess.load_image(image)
    pred_sess.run_prediction()
    pred_sess.write_annotated_image()

    return pred_sess.predictions
