# Download Repo and Dataset

git clone https://github.com/salesforce/WikiSQL
cd WikiSQL
pip install -r requirements.txt
tar xvjf data.tar.bz2

# Move sampler.py into WikiSQL
cp ../sampler.py ./
python sampler.py

