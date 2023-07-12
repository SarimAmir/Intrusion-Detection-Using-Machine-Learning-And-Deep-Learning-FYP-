# Project Title
Intrusion Detection Using Machine Learning And Deep Learning for IoT (FYP) 
# Project Description
It is an implementation of our research work. We implemented DT, RF and Bi-LSTM on ARP_MITM and SSDP FLOOD datasets. For RF and DT, concatenated dataset was divided into three ratios: 70%:30%, 80%:20%, and 90%:10% for training and testing respectively. Both models were trained on separate datasets as well with the same ratios.
For Bi-LSTM, concatenated dataset was used with different number of layers. All the models performs very well and had an accuracy over 99% but Random Forest outperform all the models. The front-end of our prototype, demonstrate the dataset upload process, which includes both malicious and normal packets. Upon uploading the dataset, the back-end model conducts predictions on the packets, accurately categorizing them as normal or malicious. The front-end then presents the performance metrics and confusion matrix, providing valuable insights into the model's effectiveness.
![Capture](https://github.com/GufranBhatti/Intrusion-Detection-Using-Machine-Learning-And-Deep-Learning-FYP-/assets/58569042/fb0a9f9c-a7e5-4b8e-a7a9-90fc91ea9940)
# Research paper
https://ieeexplore.ieee.org/document/10099152
# How to use project
1. The files are notebook files so jupyter notebook IDE or Dataspell IDE is required.
2. You can install the dataset from this link. https://www.kaggle.com/datasets/ymirsky/network-attack-dataset-kitsune
3. The front-end prototype is made using streamlit so you can run it on pycharm. Open the project in pycharm, goto terminal, run the command "streamlit run main.py"
# Credits
It's my FYP with my FYP partner Sarim Amir https://github.com/SarimAmir and my supervisor Misbah Anwer and Yumna Iftikhar.
# How to contribute to project
We only trained our model on 19GB dataset out of 64GB because of the limited resources. You can built a more powerful intrusion detection model by using the whole dataset. You can also apply different algorithms and different techniques to increase the accuracy.
