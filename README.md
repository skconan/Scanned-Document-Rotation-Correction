# Scanned Document Rotation Correction
  
  The project creates the models and service API for predicting scanned document images' angles ranging between -90° (counter clockwise) to 90° (clockwise) from the vertical.
  
## Setup Environment For Model Training and Testing 

    conda env create -f conda_env.yml -p <path of env>

    conda activate <env_name> 

    pip install -r requirements.txt

  
## Training
  
1. Dataset preparation
    
   - Convert PDF documents to images.
   
   - Download scanned document images from website.
   
   - Etc.
  
2. Data augmentation (Coming soon)
  
        python src/data_augmentation.py
    
3. Save dataset to Torch format (.pt) (Coming soon)
  
        python src/create_dataset.py
    
4. Training

   4.1 Edit dataset path
   
   4.2 Training

        python src/training.py
        
   4.3 TensorBoard Running
   
        tensorboard --logdir=<log dir> --port <port>
        
        tensorboard --logdir=runs --port 8000
        

## Models

  Download [rotation_net.onnx](https://drive.google.com/file/d/1VfDIFAZghT9qtjToOJNXG8Zk6xRZxv4T/view?usp=sharing) into models/

  <img width="50%" align="center" src="https://raw.githubusercontent.com/skconan/Scanned-Document-Rotation-Correction/main/tensorboard.png" />
  
## Running Service API with docker

  1. Start up application by running.

          docker-compose up
     
     Or
        
          docker-compose -f .\docker-compose.yml up
          
  2. Open http://127.0.0.1:9000/docs API documentation.
  
  3. Send POST Request to http://127.0.0.1:9000/angle or api_endpoint/angle with Json format.
          
          {
            "image" : "image_base_64"
          } 
          
  
  4. Response is angle in range from -90° to 90° (counter clockwise to clockwise). 
  
          {
              "angle": 0.7157974243164062
          }


          
     
  
## Testing Output
  
  | Success Cases             |  Failed Cases |
  :-------------------------:|:-------------------------:
  |<img  src="https://raw.githubusercontent.com/skconan/Scanned-Document-Rotation-Correction/main/testing_output/02.png" /> | <img src="https://raw.githubusercontent.com/skconan/Scanned-Document-Rotation-Correction/main/testing_output/00.png" /> |
  |<img  src="https://raw.githubusercontent.com/skconan/Scanned-Document-Rotation-Correction/main/testing_output/10.png" /> | <img src="https://raw.githubusercontent.com/skconan/Scanned-Document-Rotation-Correction/main/testing_output/12.png" /> |
  |<img  src="https://raw.githubusercontent.com/skconan/Scanned-Document-Rotation-Correction/main/testing_output/09.png" /> | <img src="https://raw.githubusercontent.com/skconan/Scanned-Document-Rotation-Correction/main/testing_output/13.png" /> |
  
 
## Todo list 

  - [ ] Add data_augmentation script
  
  - [ ] Add create dataset script
  
  - [ ] Impove model accuracy 


## Contributing

Anyone's participation is welcome! Open an issue or submit PRs.
