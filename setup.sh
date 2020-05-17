GREEN=$(tput setaf 2)
NORMAL=$(tput sgr0)

printf "${GREEN}Installing requirements${NORMAL}\n\n"
pip install -r src/requirements.txt

printf "${GREEN}Initializing Airflow database${NORMAL}\n\n"
airflow initdb

printf "${GREEN}Downloading dataset${NORMAL}\n\n"
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip -P ./data
tar -C data/ -xvf ./data/household_power_consumption.zip

printf "${GREEN}Preprocessing dataset${NORMAL}\n\n"
python src/preprocessing/data_preprocessing.py \
       --data_path data/household_power_consumption.txt \
       --samples_offset D \
       --history_size 15 \
       --tfrecord_path data/dataset.tfrecord

printf "${GREEN}Deleting raw dataset files${NORMAL}\n\n"
rm data/household_power_consumption.zip
rm data/household_power_consumption.txt

printf "${GREEN}Preparing Airflow DAG ${NORMAL}\n\n"
mkdir -p ~/airflow/dags
mkdir -p ~/airflow/data
cp src/pipeline.py ~/airflow/dags/
cp src/transform.py ~/airflow/dags/
cp src/trainer.py ~/airflow/dags/
cp data/dataset.tfrecord ~/airflow/data/
