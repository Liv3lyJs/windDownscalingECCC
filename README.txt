GENERAL PROJECT INFORMATION
Code from the Paper "Interpolation-free deep learning for meteorological downscaling on unaligned grids across multiple domains 
                     with application to wind power" 
currently in revision at Artificial Intelligence for the Earth Systems Journal.
This project has been done by:
                                .Jean-Sebastien Giroux 
                                                       Background: 
                                                       - Undergraduate Electrical Engineering Student at Sherbrooke University,
                                                       - Internship with Environment and Climate Change Canada (ECCC).
                                                       Role in the project:
                                                       - Lead Programmer - Data processing and Neural Netowork pipeline and optimization, 
                                                       - Deep Learning Researcher - Coming out with research ideas,
                                                       - Article Writting - Writing the first version of the paper DL.
                                                       * Corresponding author - girj2625@usherbrooke.ca
                                .Simon-Philippe Breton 
                                                       Background:
                                                       - Physical Scientist ECCC, Expert in wind and physic.
                                                       Role in the project:
                                                       - Project Leader - In charge of the downscaling project at ECCC,
                                                       - Program developper - Developped code for meteorological models, and wind ramp detection
                                                       - Meteorological Researcher - Coming out with research ideas.
                                                       - Article Writting - Writing the first version of the paper Meteorological section. 
								.Julie Carreau        
                                                       Background:
                                                       - Assistant Professor Mathematics Polytechnique Montreal.
                                                       Role in the project:                       
                                                       - Deep Learning Researcher - Coming out with research ideas,
                                                       - Mentors - Guiding the DL section, 
                                                       - Article Writting - Improving the article quality.

The project done from 10/01/2023 to 09/30/2024 (1 year), funded by Resource Naturel Canada.

The goal of the project is to use a Deep Learning (DL) algorithm to downscale wind (UV) at a 10m level from the REPS low resolution grid (10Km) 
to the Mesoanalysis High resolution grid (2.5Km) at 10m level. 


CODE INFORMATION
- Create a conda virtual environment and download packages:
    1. Install anaconda or miniconda
    2. When installation done do the following line:
        conda create --name myenv python=3.8
        conda activate myenv
        pip install -r requirements.txt

- How does the code work?
    1. Data_Processing
    - In data processing, it is where the data is processed before being passed to the Neural Network. 
            A. A00_calculate_gradients
                Here we are doing the first step for the data processing which is finding gradient value. 

                In this script we are getting uu, vv, tt and uv gradients. Note that if you want to add another
                gradient calculation it's totally possible to do so. You will need to modify the appropriate code 
                section to allow your changes. Note that the code has been done quickly and need to be optimize later.

                1.0 Run calculate_gradient_uv.py
                - This script will calculate the UV value for each file for the REPS. 
                - These values are not in the REPS original data so they need to be calculated with the help of pgsm
                https://wiki.cmc.ec.gc.ca/wiki/RPN-SI/RpnUtilities/pgsm 
                https://wiki.cmc.ec.gc.ca/wiki/RPN-SI/RpnUtilities/pgsm/references/GRILLE 
                - In order to run the script use VScode and put the appropriate constantes in the constant section and 
                once it is done it will automatically execute by itserf. 
                - This script will output the uv value for each standard file in the output folder. It will calculate the UV
                value and output GZ, P0 and UV value in a standard file later used to calculate the gradients.

                2.0 Run calculate_gradient.py
                - This script will calculate the gradient for uu, vv, tt and uv variable. 
                - Specifies the path that the original data are located for uv and for the other input variables
                - Specifies the output path of uu, vv, tt and uv variable.
                - You also need to specify the level in which the gradients are going to be calculated. 
                - The output file will be the gradient calculated for each variables with P0 values that will
                be deleted later. 

                3.0 Run combine_std.py
                - This script will combine standard file with gradients and original value to allow standard files to have both 
                values in it. It will also delete P0 values that are unnecessary!
                - This scirpt is the last step for A00 calculate gradient. Once it is done you can now move to A0 to extract hourly data.
            
            B. A0_24to1hours_reps
                This is Simon-Philippe script to put 24h data on a hourly file. 

                - You have to select the start date and the end date
                - You also have to select the input folder and outpur folder
                - You need to select the model pass. In can be either 00, 06, 12, 18. 

                - To run the script do the modifications and in the terminal do:
                ./REPS_24h_to_1h.sh

            C. A_crop_interpolate
            D. B_convert_npy
            E. C_normalize
            Bonus. data_processing.sh


    2. Neural Network

