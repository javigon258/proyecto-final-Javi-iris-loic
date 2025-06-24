# I create my virtual environment to work better and avoid mixing library versions with the Python installed locally.

My virtual environment is: .\env\Scripts\activate.bat

pip install -r requirements.txt

Installed libraries in notebook: python -m spacy download en_core_web_sm

# Using Git LFS

Git LFS allows you to upload large files,  
but it requires special configuration  
and users who clone the repo must also install it.

git lfs install

git lfs track "*.csv"

git lfs track "*.pkl"

git add .gitattributes

git add data/*.csv models/*.pkl

git commit -m "Add large files with Git LFS"

git push origin main