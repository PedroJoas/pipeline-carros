from flask import Flask, render_template, request
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
MODELO_DIR = BASE_DIR / 'models'
DADOS_PATH = BASE_DIR / 'dados' / 'carros_transformado.csv'

df = pd.read_csv(DADOS_PATH)
modelo_path = next(MODELO_DIR.glob('*.pkl'))
modelo = joblib.load(modelo_path)

colunas_categoricas = [
    "Marca", "Tipo de veículo", "Potência do motor", 
    "Combustível", "Câmbio", "Direção", 
    "Portas", "Tipo de direção"
]

opcoes_formulario = {
    coluna: sorted(df[coluna].dropna().unique()) for coluna in colunas_categoricas
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        entrada = {col: request.form[col] for col in colunas_categoricas}
        entrada["Ano"] = int(request.form["Ano"])
        entrada["Quilometragem"] = float(request.form["Quilometragem"])
        entrada_df = pd.DataFrame([entrada])
        preco = modelo.predict(entrada_df)[0]
        return render_template('form.html', opcoes=opcoes_formulario, resultado=preco)
    
    return render_template('form.html', opcoes=opcoes_formulario)

if __name__ == '__main__':
    app.run(debug=True)
