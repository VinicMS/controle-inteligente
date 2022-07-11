import numpy as np              # operações matemátcas
import matplotlib.pyplot as plt # plot de gráficos
import pandas as pd             # manipulação de arquivos (csv, txt)
import joblib                   # salvar rede
import pathlib                  #contar arquivos para nomear automaticamente

from sklearn.model_selection import train_test_split #dividir os dados
from sklearn.metrics import r2_score, mean_squared_error #validar rede
from sklearn.neural_network import MLPRegressor

titulos_coleta = np.array([
    "erro_posicao",
    "x_p",  
    "theta",
    "theta_p",
    "acao"
])

dataframe = pd.read_csv(
    'dados/dados_gerados.csv',
    header = None,
    names = titulos_coleta)

entradas = dataframe.iloc[:, :4]
saida = dataframe.iloc[:, 4]

entradas_train, entradas_test, saida_train, saida_test = train_test_split(
    entradas,
    saida,
    test_size = 0.33)

#13
#[128, 128, 128]
#relu
#batch_size=500
#max_iter = 100000
#solver = 'adam'
#verbose = True
#learning_rate = 'adaptative'

modelo = MLPRegressor(
    hidden_layer_sizes = [64, 32, 1],
    activation='relu',
    #batch_size = 1000,
    max_iter = 100000,
    solver='adam',
    verbose = True,
    learning_rate = 'constant',
    # tol = 0.00001,
    n_iter_no_change = 50,
    shuffle = True
)

modelo.fit(entradas_train, saida_train)

saida_pred = modelo.predict(entradas_test)

erro = mean_squared_error(saida_test, saida_pred)

print("O erro medio quadratico é:  %.2f" % (erro))

medidaR2 = r2_score(saida_test, saida_pred)
print("A medida R2 é: %.2f" % (medidaR2))

#criar o arquivo gerado com um numero novo (numero de arquivos do diretorio+1
#n_arq+1
n_arq = 0
for path in pathlib.Path("modelos").iterdir():
    if path.is_file():
        n_arq += 1

identificador_arquivo = (n_arq+1)

arquivo = "tr" + str(identificador_arquivo)
print("------------------------------ \n\nModelo salvo como: " + arquivo + "\n\n ------------------------------")
joblib.dump(modelo, "modelos/"+arquivo)

plt.figure(1, figsize=(10, 10))

plt.scatter(saida_pred, saida_test)
plt.plot(saida_pred, saida_pred, color="red")
plt.title("Modelo " + str(identificador_arquivo))
# plt.xlim([-2000, 2000])
# plt.ylim([-2000, 2000])
plt.grid()
plt.show()