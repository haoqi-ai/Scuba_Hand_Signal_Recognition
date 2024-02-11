import cv2
import numpy as np
import random
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog, messagebox


class ImageHandler:
    def __init__(self, min_area, min_length, distance, draw_type=0, max_area=100000, max_length=100000):
        """
        Initialize the image handler with specific parameters for image processing.
        :param min_area: Minimum area for contour detection.
        :param min_length: Minimum perimeter length for contour detection.
        :param distance: Distance threshold for point detection.
        :param draw_type: Type of drawing to apply on detected contours (0 for polyline, 1 for lines).
        :param max_area: Maximum area for contour detection.
        :param max_length: Maximum perimeter length for contour detection.
        """
        self.min_area = min_area
        self.max_area = max_area
        self.min_length = min_length
        self.max_length = max_length
        self.distance = distance
        self.points_list = []  # List to store points of detected contours
        self.high_HSV = np.array([15, 255, 255])  # Upper HSV threshold for filtering
        self.low_HSV = np.array([0, 50, 50])  # Lower HSV threshold for filtering
        self.draw_type = draw_type
        self.img = None
        self.output_img = None

    def change_distance(self, distance):
        """
        Update the distance threshold for point detection.
        :param distance: New distance threshold.
        """
        self.distance = distance

    def change_HSV(self, low_HSV, high_HSV):
        """
        Update the HSV color filtering bounds.
        :param low_HSV: New lower HSV bound.
        :param high_HSV: New upper HSV bound.
        """
        self.low_HSV = low_HSV
        self.high_HSV = high_HSV

    def resize_img(self, img, target_size=(480, 640, 3)):
        """
        Resize the image to a specified size.
        :param img: Image to resize.
        :param target_size: Target size as a tuple (height, width, channels).
        :return: Resized image.
        """
        if img is not None:
            self.img = img
            h, w = img.shape[:2]
            scale = min(target_size[1] / w, target_size[0] / h)
            resized_img = cv2.resize(img, None, fx=scale, fy=scale)
            mask = np.zeros(target_size, dtype=np.uint8)
            new_h, new_w = resized_img.shape[:2]
            x_offset = (target_size[1] - new_w) // 2
            y_offset = (target_size[0] - new_h) // 2
            mask[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = resized_img
            return mask
        return img  # Return the original image if input is None

    def img_handle(self, img=None):
        """
        Process the image for contour detection.
        :param img: Image to process.
        :return: Processed image.
        """
        if img is not None:
            self.img = img

        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        cv2.GaussianBlur(self.img, (5, 5), 0)
        self.img = cv2.inRange(self.img, self.low_HSV, self.high_HSV)
        kernel = np.ones((3, 3), np.uint8)
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel)
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(self.img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            length = cv2.arcLength(contour, True)
            if self.max_area > area > self.min_area and self.max_length > length > self.min_length:
                epsilon = 0.02 * length
                approx_points = cv2.approxPolyDP(contour, epsilon, True)
                approx_points = approx_points.reshape(len(approx_points), 2)
                self.points_list = np.array(approx_points, dtype=np.int32)
                if self.draw_type == 0:
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    cv2.polylines(self.output_img, [self.points_list], True, color, 4)
        return self.img

    def get_distance(self, pt1, pt2):
        """
        Calculate the Euclidean distance between two points.
        :param pt1: First point as a tuple (x, y).
        :param pt2: Second point as a tuple (x, y).
        :return: Euclidean distance.
        """
        return ((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2) ** 0.5

    def detect(self):
        """
        Detect specific gestures based on the processed contours.
        :return: Gesture detected as a string.
        """
        num = 0
        if self.points_list.any():
            max_index = np.argmax(self.points_list, axis=0)
            for point in self.points_list:
                distance = self.get_distance(self.points_list[max_index[1]], point)
                if distance > self.distance:
                    if self.draw_type == 1:
                        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                        cv2.line(self.output_img, self.points_list[max_index[1]], point, color, 4)
                    num += 1
            if num == 1:
                gesture = 'One'
            elif num == 2:
                gesture = 'Scissors'
            elif num == 3:
                gesture = 'OK'
            elif num == 4:
                gesture = 'Four'
            elif num == 5:
                gesture = 'Paper'
            else:
                gesture = 'Unknown'
            cv2.putText(self.output_img, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            return gesture

    def get_hand(self, img):
        """
        Main function to handle hand gesture detection.
        :param img: Image to process.
        :return: Tuple of the original image, processed image, and detected gesture.
        """
        self.img = img
        if img.shape[:2] != (480, 640):
            self.img = self.resize_img(img, (480, 640, 3))
        self.output_img = np.copy(self.img)
        self.img_handle()
        gesture = self.detect()
        return self.output_img, self.img, gesture


class GestureRecognitionGUI:
    def __init__(self):
        self.image_handler = ImageHandler(20000, 1000, 280)
        self.result_text = ''
        self.video_capture = None
        self.after_id = None
        self.supported_files = ['.mp4', '.png', '.jpg']
        self.selected_file = ''

        self.root = tk.Tk()
        self.root.geometry('1000x700')
        self.root.title('Gesture Recognition')
        self.root.resizable(width=False, height=False)

        self.img1_label = tk.Label(self.root, text='', bg='white', bd=10)
        self.img1_label.place(x=340, y=20, width=640, height=480)

        self.img2_label = tk.Label(self.root, text='', bg='white', bd=10)
        self.img2_label.place(x=20, y=220, width=250, height=400)

        self.select_file_button = tk.Button(self.root, text='Select File', command=self.select_file, font=('Arial', 20), bg='green', bd=10)
        self.select_file_button.place(x=20, y=20, width=250, height=50)

        self.open_file_button = tk.Button(self.root, text='Open', command=self.open_file, font=('Arial', 20), bg='blue', bd=10)
        self.open_file_button.place(x=20, y=90, width=250, height=50)

        self.open_camera_button = tk.Button(self.root, text='Open Camera', command=self.toggle_camera, font=('Arial', 20), bg='white', bd=10)
        self.open_camera_button.place(x=20, y=160, width=250, height=50)

        self.result_var = tk.StringVar(self.root, value='')
        self.result_entry = tk.Entry(self.root, textvariable=self.result_var, state='readonly', font=('Arial', 38), bg='white', bd=10)
        self.result_entry.place(x=340, y=520, width=640, height=140)

        self.distance_threshold_var = tk.IntVar(self.root)
        self.distance_scale = tk.Scale(self.root, label='Distance Threshold', from_=0, to=800, resolution=1, orient=tk.HORIZONTAL, tickinterval=200, variable=self.distance_threshold_var, bg='white', bd=10)
        self.distance_scale.place(x=20, y=620, width=250)
        self.update_result()

    def toggle_camera(self):
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
            if self.after_id:
                self.root.after_cancel(self.after_id)
        else:
            self.video_capture = cv2.VideoCapture(0)
            self.stream_video()

    def stream_video(self):
        success, img = self.video_capture.read()
        if success and img is not None:
            processed_img1, processed_img2, gesture = self.image_handler.get_hand(img)
            self.display_image1(processed_img1)
            self.display_image2(processed_img2)
        self.after_id = self.root.after(10, self.stream_video)

    def open_file(self):
        if not self.selected_file:
            messagebox.showerror(title='Warning', message='Please select a video or image file.')
        else:
            if any(ext in self.selected_file for ext in self.supported_files):
                if self.video_capture:
                    self.video_capture.release()
                self.video_capture = cv2.VideoCapture(self.selected_file)
                if self.after_id:
                    self.root.after_cancel(self.after_id)
                self.stream_video()
            else:
                if self.video_capture:
                    self.video_capture.release()
                img = cv2.imread(self.selected_file)
                processed_img1, processed_img2, gesture = self.image_handler.get_hand(img)
                self.display_image1(processed_img1)
                self.display_image2(processed_img2)

    def select_file(self):
        self.selected_file = filedialog.askopenfilename()
        if not any(ext in self.selected_file for ext in self.supported_files):
            messagebox.showerror(title='Warning', message='Please select a video or image file.')

    def update_result(self):
        distance = self.distance_threshold_var.get()
        self.image_handler.change_distance(distance)
        self.result_var.set(f'Gesture detection result: {self.result_text}')
        self.root.after(10, self.update_result)

    def display_image1(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.img1_label.image = img_tk
        self.img1_label['image'] = img_tk

    def display_image2(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = self.image_handler.resize_img(img, target_size=(400, 250, 3))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.img2_label.image = img_tk
        self.img2_label['image'] = img_tk

    def run(self):
        messagebox.showinfo('Notice', message='Adjust the distance threshold slider to around 273 before starting, and wear a mask for better detection accuracy.')
        self.root.mainloop()

    def close(self):
        if self.video_capture:
            self.video_capture.release()


if __name__ == "__main__":
    app = GestureRecognitionGUI()
    app.run()
    app.close()
