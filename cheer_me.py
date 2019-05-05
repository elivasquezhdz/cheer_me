__author__ = 'elfx'
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from get_data import get_data
import cv2
import numpy as np
import tensorflow as tf
import datetime as dt
from random import shuffle
import os

class CamApp(App):

    def get_images(self):
        images = get_data()
        return images

    def get_meme(self):
        shuffle(self.images)
        folder,image = self.images[0]
        image_name = os.path.join("images",folder,image)
        img = cv2.imread(image_name)
        return (folder,image,img)

    def load_labels(self,label_file):
        label = []
        proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())
        return label
    def load_graph(self,model_file):
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()

        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)

        return graph
    def read_tensor_from_image_variable(self,img, input_height=299, input_width=299,
                    input_mean=0, input_std=255):
        float_caster = tf.cast(img, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0);
        resized = tf.image.resize(dims_expander, [input_height, input_width], method=tf.image.ResizeMethod.BILINEAR)
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        sess = tf.compat.v1.Session()
        result = sess.run(normalized)
        return result  

    def set_graph(self):
        self.input_height = 192
        self.input_width = 192
        self.input_mean = 128
        self.input_std = 128
        input_layer = "input"
        output_layer = "final_result"
        model_file = "graph.pb"
        label_file = "labels.txt"
        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        graph = self.load_graph(model_file)
        self.input_operation = graph.get_operation_by_name(input_name);
        self.output_operation = graph.get_operation_by_name(output_name);
        self. labels = self.load_labels(label_file)
        return graph

    def date_string(self):
        today = dt.datetime.today()
        year = today.year
        mont = today.month
        day = today.day
        hour = today.hour
        minu = today.minute
        sec = today.second
        dstring = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(year,mont,day,hour,minu,sec)
        return dstring

    def write_log(self,message):
        with open("emotions.log","a") as log:
            log.write(message + "\n")
    def build(self):
        self.graph = self.set_graph()
        self.frame_count = 0
        self.images = self.get_images()
        self.img1=Image()
        layout = BoxLayout()
        layout.add_widget(self.img1)
        #opencv2 stuffs
        self.capture = cv2.VideoCapture(0)
        #cv2.namedWindow("CV2 Image")
        Clock.schedule_interval(self.update, 1.0/30.0)

        return layout

    def update(self, dt):
        # display image from cam in opencv window
        if(self.frame_count==0):
             self.meme = self.get_meme()
        if(self.frame_count%5==0):
              self.meme = self.get_meme()
        self.frame_count += 1
        ret, frame = self.capture.read()
        with tf.compat.v1.Session(graph=self.graph) as sess:
            t = self.read_tensor_from_image_variable(frame,
                                    input_height=self.input_height,
                                    input_width=self.input_width,
                                    input_mean=self.input_mean,
                                    input_std=self.input_std)
            results = sess.run(self.output_operation.outputs[0],
                        {self.input_operation.outputs[0]: t})
            results = np.squeeze(results)
            top_k = results.argsort()[-5:][::-1]
            for i in top_k:
                human_string = "{},".format(self.labels[i]) + "{0:2f}".format(results[i])
                #print(human_string)
                date_log = self.date_string()
                self.write_log("{}:{}:{}:{}".format(date_log,human_string,self.meme[0],self.meme[1]))
                break
            #black = np.zeros(frame.shape,dtype="uint8")
     
            #blackframe = cv2.putText(self.meme , human_string, (20, 400), cv2.FONT_HERSHEY_TRIPLEX, 2.5, (0, 255, 0))    

        # convert it to texture
        buf1 = cv2.flip( self.meme[-1], 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=( self.meme[-1].shape[1],  self.meme[-1].shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.img1.texture = texture1

if __name__ == '__main__':
    CamApp().run()
    cv2.destroyAllWindows()