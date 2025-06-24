# Creo mi entorno virtual para trabajar mejor y evitar mezclar versiones de librerias con python que está ne local. 

Mi entorno virtual es: .\env\Scripts\activate.bat

pip install -r requirements.txt

# Usar Git LFS

Git LFS permite subir archivos grandes, 
pero requiere configuración especial
y los usuarios que clonen el repo también deben instalarlo

git lfs install
git lfs track "*.csv"
git lfs track "*.pkl"
git add .gitattributes
git add data/*.csv models/*.pkl
git commit -m "Agregar archivos grandes con Git LFS"
git push origin main