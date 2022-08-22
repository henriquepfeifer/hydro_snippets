from select import select
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

def remove_outliers(x_, y_, threshold=3.5, steps=2):
    '''Median Absolute Deviation (MAD) based outlier detection'''
    filtered_x = pd.DataFrame()
    filtered_y = pd.DataFrame()
    max_x = max(x_.values)
    min_x = min(x_.values)
    x_step = (max_x - min_x)/steps

    for i in range(1, steps + 1):
        x_start = min_x + x_step * (i - 1)
        x_end = min_x + x_step * i

        mask = x_.between(x_start, x_end).values
        y = y_.iloc[mask]

        median = np.median(y, axis=0)
        #diff = np.sum((y.values - median) ** 2, axis=-1)
        diff = (y.values - median) ** 2
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)
        # scale constant 0.6745
        modified_z_score = 0.6745 * diff / med_abs_deviation
        y[modified_z_score > threshold] = np.nan
        y = y.dropna()
        x = x_.loc[y.index]
        filtered_y = pd.concat([filtered_y, y])
        filtered_x = pd.concat([filtered_x, x])

    return filtered_x, filtered_y 

def vazao_comporta_pop(abertura1, abertura2, nv_montante):
    vazao_comporta = pd.DataFrame(0, index=abertura1.index)

    fundo = 807.3

    # necessário colocar tolerância de 0.1 para não aparecer vazão quando as comportas estão totalmente fechadas pois não há precisão nas leituras
    index_abertura1 = abertura1 > 0.1
    index_abertura2 = abertura2 > 0.1

    vazao_comporta[index_abertura1] = (abertura1[index_abertura1]/100) * 4.7 * (9.79 * 2 * (nv_montante[index_abertura1] - fundo - abertura1[index_abertura1] / 200)) ** (1/2) * 0.67
    vazao_comporta[index_abertura2] = vazao_comporta[index_abertura2] + (abertura2[index_abertura2]/100) * 4.7 * (9.79 * 2 * (nv_montante[index_abertura2] - fundo - abertura2[index_abertura2] / 200)) ** (1/2) * 0.67

    return vazao_comporta

def vazao_vertida_pop(nv_montante):
    vazao = pd.Series(0, index=nv_montante.index)
    c_c_zero = pd.Series(0, index=nv_montante.index)
    c_c_zero_taipas = pd.Series(0, index=nv_montante.index)
    d_nv = pd.Series(0, index=nv_montante.index)
    d_nv_taipas = pd.Series(0, index=nv_montante.index)
    
    # constantes do vertedouro
    nv_soleira = 817
    h_projeto = 2.69
    c_zero = 2.2
    l_util = 105.11

    # constantes das taipas
    nv_taipas = 817.70
    l_util_taipas = 49.82
    c_zero_taipas = 2
    h_projeto_taipas = 2


    index_vertedouro = nv_montante > nv_soleira
    index_taipas = nv_montante > nv_taipas

    d_nv.loc[index_vertedouro] = nv_montante.loc[index_vertedouro] - nv_soleira
    d_nv_taipas.loc[index_taipas] = nv_montante.loc[index_taipas] - nv_taipas
    
    c_c_zero.loc[index_vertedouro] = 0.096006 * (d_nv.loc[index_vertedouro]/h_projeto) ** 3 - 0.27062 * (d_nv.loc[index_vertedouro]/h_projeto) ** 2 + 0.386699 * (d_nv.loc[index_vertedouro]/h_projeto) + 0.783742
    c_c_zero_taipas.loc[index_taipas] = 0.096006 * (d_nv_taipas.loc[index_taipas]/h_projeto_taipas) ** 3 - 0.27062 * (d_nv_taipas.loc[index_taipas]/h_projeto_taipas) ** 2 + 0.386699 * (d_nv_taipas.loc[index_taipas]/h_projeto_taipas) + 0.783742

    vazao.loc[index_vertedouro] = c_zero * c_c_zero.loc[index_vertedouro].multiply(d_nv.loc[index_vertedouro].pow(3/2)) * l_util
    vazao.loc[index_taipas] = vazao.loc[index_taipas] + c_zero_taipas * c_c_zero_taipas.loc[index_taipas].multiply(d_nv_taipas.loc[index_taipas].pow(3/2)) * l_util_taipas

    return vazao


def determinar_q_ugs(potencias, niveis, fit_potencia, fit_gerador, erro_admissivel):
    
    #determinar vazão inicial
    p_eixo_ug1 = np.divide(potencias['UG1'], pot_eixo(potencias['UG1'], fit_gerador[0][0], fit_gerador[0][1], fit_gerador[0][2]))
    p_eixo_ug2 = np.divide(potencias['UG2'], pot_eixo(potencias['UG2'], fit_gerador[0][0], fit_gerador[0][1], fit_gerador[0][2]))
    p_eixo_ug3 = np.divide(potencias['UG3'], pot_eixo(potencias['UG3'], fit_gerador[0][0], fit_gerador[0][1], fit_gerador[0][2]))
    p_eixo_ug4 = np.divide(potencias['UG4'], pot_eixo(potencias['UG4'], fit_gerador[0][0], fit_gerador[0][1], fit_gerador[0][2]))

    q_ug1 = vazao_p_h(pd.concat([p_eixo_ug1, niveis['D_NV_UG1']], axis=1).values, fit_potencia[0][0], fit_potencia[0][1], fit_potencia[0][2], fit_potencia[0][3], fit_potencia[0][4])
    q_ug2 = vazao_p_h(pd.concat([p_eixo_ug2, niveis['D_NV_UG2']], axis=1).values, fit_potencia[0][0], fit_potencia[0][1], fit_potencia[0][2], fit_potencia[0][3], fit_potencia[0][4])
    q_ug3 = vazao_p_h(pd.concat([p_eixo_ug3, niveis['D_NV_UG3']], axis=1).values, fit_potencia[0][0], fit_potencia[0][1], fit_potencia[0][2], fit_potencia[0][3], fit_potencia[0][4])
    q_ug4 = vazao_p_h(pd.concat([p_eixo_ug4, niveis['D_NV_UG4']], axis=1).values, fit_potencia[0][0], fit_potencia[0][1], fit_potencia[0][2], fit_potencia[0][3], fit_potencia[0][4])


    q_ug1[potencias['UG1']==0] = 0
    q_ug1 = pd.Series(q_ug1, index=p_eixo_ug1.index)

    q_ug2[potencias['UG2']==0] = 0
    q_ug2 = pd.Series(q_ug2, index=p_eixo_ug2.index)

    q_ug3[potencias['UG3']==0] = 0
    q_ug3 = pd.Series(q_ug3, index=p_eixo_ug3.index)

    q_ug4[potencias['UG4']==0] = 0
    q_ug4 = pd.Series(q_ug4, index=p_eixo_ug4.index)

    erro = pd.Series(2 * erro_admissivel, index=p_eixo_ug1.index)

    for selected_index in p_eixo_ug1.index:
        while erro[selected_index] >= erro_admissivel:
            # print("selected index= ", selected_index)
            q_total = q_ug1[selected_index] + q_ug2[selected_index] + q_ug3[selected_index] + q_ug4[selected_index]

            # determinar perda de carga
            dh_1 = 2.293483 * 10 ** -4 * q_ug1[selected_index] ** 2 + 2.689919 * 10 ** -6 * q_total ** 2
            dh_2 = 2.293483 * 10 ** -4 * q_ug2[selected_index] ** 2 + 2.689919 * 10 ** -6 * q_total ** 2
            dh_3 = 2.293483 * 10 ** -4 * q_ug3[selected_index] ** 2 + 2.689919 * 10 ** -6 * q_total ** 2
            dh_4 = 2.293483 * 10 ** -4 * q_ug4[selected_index] ** 2 + 2.689919 * 10 ** -6 * q_total ** 2

            # Entradas de vazao_p_h
            # potencia de eixo, nova queda líquida
            in_ug1 = [p_eixo_ug1[selected_index], niveis['D_NV_UG1'][selected_index] - dh_1]
            in_ug2 = [p_eixo_ug2[selected_index], niveis['D_NV_UG2'][selected_index] - dh_2]
            in_ug3 = [p_eixo_ug3[selected_index], niveis['D_NV_UG3'][selected_index] - dh_3]
            in_ug4 = [p_eixo_ug4[selected_index], niveis['D_NV_UG4'][selected_index] - dh_4]
            
            # determinar vazão com queda líquida
            q_ug1_recalculado = vazao_p_h(in_ug1, fit_potencia[0][0], fit_potencia[0][1], fit_potencia[0][2], fit_potencia[0][3], fit_potencia[0][4])
            q_ug2_recalculado = vazao_p_h(in_ug2, fit_potencia[0][0], fit_potencia[0][1], fit_potencia[0][2], fit_potencia[0][3], fit_potencia[0][4])
            q_ug3_recalculado = vazao_p_h(in_ug3, fit_potencia[0][0], fit_potencia[0][1], fit_potencia[0][2], fit_potencia[0][3], fit_potencia[0][4])
            q_ug4_recalculado = vazao_p_h(in_ug4, fit_potencia[0][0], fit_potencia[0][1], fit_potencia[0][2], fit_potencia[0][3], fit_potencia[0][4])

            # queda líquida é determinada de acordo com a vazão das máquinas
            erro_ug1 = q_ug1_recalculado - q_ug1[selected_index]
            erro_ug2 = q_ug2_recalculado - q_ug2[selected_index]
            erro_ug3 = q_ug3_recalculado - q_ug3[selected_index]
            erro_ug4 = q_ug4_recalculado - q_ug4[selected_index]

            erro[selected_index] = max(abs(erro_ug1), abs(erro_ug2), abs(erro_ug3), abs(erro_ug4))
            # ajustar variáveis para próxima iteração
            q_ug1[selected_index] = q_ug1_recalculado
            q_ug2[selected_index] = q_ug2_recalculado
            q_ug3[selected_index] = q_ug3_recalculado
            q_ug4[selected_index] = q_ug4_recalculado

    vazoes_df = pd.concat([q_ug1, q_ug2, q_ug3, q_ug4], axis=1)
    vazoes_df = vazoes_df.rename(columns={0:'UG1', 1:'UG2', 2:'UG3', 3:'UG4'})

    return vazoes_df

def vazao_p_h(data, a, b, c, d, e):
    if len(data)>2:
        fx = a * data[:, 0] ** 2 + b * data[:, 0] + c
        gx = d * fx * data[:, 1] + e
    else:
        if data[0] == 0:
            gx = 0
        else:
            fx = a * data[0] ** 2 + b * data[0] + c
            gx = d * fx * data[1] + e
    return gx

def pot_eixo(data, a, b, c):
    fx = a * data ** 2 + b * data + c
    return fx

def curva_chave(vazao, a, b, c):
    nivel = c * pow(vazao, a) + b
    return nivel

# importar dados da usina
p_ug1 = pd.read_csv('ferramentas_hidro/curva_chave_jusante_pop/DADOS/GRANDEZASUG01_DATA_TABLE.csv').sort_index()
p_ug2 = pd.read_csv('ferramentas_hidro/curva_chave_jusante_pop/DADOS/GRANDEZASUG02_DATA_TABLE.csv').sort_index()
p_ug3 = pd.read_csv('ferramentas_hidro/curva_chave_jusante_pop/DADOS/GRANDEZASUG03_DATA_TABLE.csv').sort_index()
p_ug4 = pd.read_csv('ferramentas_hidro/curva_chave_jusante_pop/DADOS/GRANDEZASUG04_DATA_TABLE.csv').sort_index()

p_ug1 = p_ug1.rename(columns={'POTENCIAATIVA':'UG1'})
p_ug2 = p_ug2.rename(columns={'POTENCIAATIVA':'UG2'})
p_ug3 = p_ug3.rename(columns={'POTENCIAATIVA':'UG3'})
p_ug4 = p_ug4.rename(columns={'POTENCIAATIVA':'UG4'})

potencias = pd.concat([p_ug1['UG1'], p_ug2['UG2'], p_ug3['UG3'], p_ug4['UG4']], axis=1)
potencias.index = pd.to_datetime(potencias.index, format='%d/%m/%y %H:%M:%S', errors='ignore')

print("p_ug1 = ", p_ug1.iloc[0,1])
print("p_ug1.index = ", p_ug1.index[0])

print("p_ug2 = ", p_ug2.iloc[0,1])
print("p_ug2.index = ", p_ug2.index[0])

print("p_ug3 = ", p_ug3.iloc[0,1])
print("p_ug3.index = ", p_ug3.index[0])

print("p_ug4 = ", p_ug4.iloc[0,1])
print("p_ug4.index = ", p_ug4.index[0])

print("potencias= ", potencias.iloc[0])

del p_ug1, p_ug2, p_ug3, p_ug4

niveis = pd.read_csv('ferramentas_hidro/curva_chave_jusante_pop/DADOS/GRANDEZASUSINA_DATA_TABLE.csv', delimiter=';')

niveis = niveis.set_index('E3TIMESTAMP', drop=True)
niveis.index = pd.to_datetime(niveis.index, format='%d/%m/%y %H:%M:%S.%f', errors='ignore').round("min")

d_nv_ug1 = niveis['POSGRADE_UG01'] - niveis['NIVELJUSANTE']
d_nv_ug1.name = 'D_NV_UG1'
d_nv_ug2 = niveis['POSGRADE_UG02'] - niveis['NIVELJUSANTE']
d_nv_ug2.name = 'D_NV_UG2'
d_nv_ug3 = niveis['POSGRADE_UG03'] - niveis['NIVELJUSANTE']
d_nv_ug3.name = 'D_NV_UG3'
d_nv_ug4 = niveis['POSGRADE_UG04'] - niveis['NIVELJUSANTE']
d_nv_ug4.name = 'D_NV_UG4'

niveis = pd.concat([niveis, d_nv_ug1, d_nv_ug2, d_nv_ug3, d_nv_ug4], axis=1)

niveis = niveis.dropna()

'''vazoes_adufas = pd.read_csv('ferramentas_hidro/curva_chave_jusante_pop/DADOS/VAZAO_ADUFAS.csv', delimiter=',')
vazoes_adufas = niveis.set_index('E3TIMESTAMP', drop=True)
vazoes_adufas.index = pd.to_datetime(vazoes_adufas.index, format='%d/%m/%y %H:%M:%S.%f', errors='ignore').round("min")'''

'''vazoes_adufas = pd.read_excel('ferramentas_hidro/curva_chave_jusante_pop/DADOS/h. Geração Elétrica_Rev19_enviado Aneel.xlsx')
vazoes_adufas = vazoes_adufas.set_index('E3TimeStamp', drop=True)
vazoes_adufas_ug1 = vazoes_adufas.loc[vazoes_adufas['Unidade Geradora']=='UG1']
vazoes_adufas_ug2 = vazoes_adufas.loc[vazoes_adufas['Unidade Geradora']=='UG2']
vazoes_adufas_ug3 = vazoes_adufas.loc[vazoes_adufas['Unidade Geradora']=='UG3']
vazoes_adufas_ug4 = vazoes_adufas.loc[vazoes_adufas['Unidade Geradora']=='UG4']

vazoes_adufas_saida = vazoes_adufas_ug1['Vazão Afluente (m³/s)'] - vazoes_adufas_ug1['Vazao Turbinada (m³/s)'] - vazoes_adufas_ug2['Vazao Turbinada (m³/s)'] - vazoes_adufas_ug3['Vazao Turbinada (m³/s)'] - vazoes_adufas_ug4['Vazao Turbinada (m³/s)'] - vazoes_adufas_ug4['Vazão vertida (m³/s)']
vazoes_adufas_saida.index = vazoes_adufas_saida.index.round('S')'''

vazoes_adufas = pd.read_csv('ferramentas_hidro/curva_chave_jusante_pop/DADOS/extracao_info_diario_pop.csv')
vazoes_adufas['E3TimeStamp'] = pd.to_datetime(vazoes_adufas['E3TimeStamp'], format='%Y/%m/%d %H:%M:%S')
vazoes_adufas = vazoes_adufas.set_index('E3TimeStamp', drop=True)

vazoes_adufas = vazoes_adufas['Vazão C1 (m³/s)'] + vazoes_adufas['Vazão C2 (m³/s)']

# vazoes_adufas.index = vazoes_adufas.index.round("S")

rendimento_turbina = pd.DataFrame({'Vazão': [57.09, 51.9, 46.71, 41.52, 36.33, 31.14, 25.95, 20.76, 15.57,
                                             57.09, 51.9, 46.71, 41.52, 36.33, 31.14, 25.95, 20.76, 15.57,
                                             57.09, 51.9, 46.71, 41.52, 36.33, 31.14, 25.95, 20.76, 15.57],
                                   'Rendimento': [.917, .915, .911, .905, .894, .879, .857, .825, .779,
                                                  .917, .918, .916, .913, .907, .896, .879, .850, .807,
                                                  .923, .924, .922, .919, .913, .903, .885, .857, .814],
                                   'Queda Líquida': [10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00,
                                                     12.67, 12.67, 12.67, 12.67, 12.67, 12.67, 12.67, 12.67, 12.67,
                                                     15.12, 15.12, 15.12, 15.12, 15.12, 15.12, 15.12, 15.12, 15.12]})

# --- 1. Transformar rendimento em potência (na curva de rendimento da turbina)
# Aceleração da gravidade [m/s²]
g = 9.80665
# massa específica da água [kg/m³]
ro = 1000

potencia_eixo = np.multiply(np.multiply(rendimento_turbina['Vazão'], rendimento_turbina['Rendimento']), rendimento_turbina['Queda Líquida']) * g * ro
potencia_eixo = pd.Series(potencia_eixo, name='Potência Eixo')
rendimento_turbina = pd.concat([rendimento_turbina, potencia_eixo], axis=1)

# --- 2. Dados de entrada de rendimento do gerador
rendimento_gerador = pd.DataFrame({'Potência': [3190, 4785, 6380], 'Rendimento': [.965, .970, .967]})

# --- 3. Curve fitting da vazão da turbina
print("Curve fitting da vazão da turbina")
fit_q = curve_fit(vazao_p_h, np.array([rendimento_turbina['Queda Líquida'].values, rendimento_turbina['Potência Eixo'].values]).T, rendimento_turbina['Vazão'].values)

# --- 4. Curve fitting do rendimento do gerador
print("Curve fitting do rendimento do gerador")
fit_p = curve_fit(pot_eixo, rendimento_gerador['Potência'].values, rendimento_gerador['Rendimento'].values)

# --- 5. Determinar vazão

# No processo iterativo, é necessário percorrer todos os valores do dataset
# Cria vetor com datas que possuem todos os dados necessários para a iteração:
potencias = potencias.sort_index()
vazoes_adufas_saida = vazoes_adufas.sort_index()
niveis = niveis.sort_index()

indices = potencias.index.intersection(niveis.index).intersection(vazoes_adufas.index)
potencias = potencias.loc[indices]
niveis = niveis.loc[indices]

# Início do processo iterativo para determinação das vazões
# erro admissível é em m³/s
vazoes_maquinas = determinar_q_ugs(potencias, niveis, fit_q, fit_p, erro_admissivel=0.1)

# Vazão Vertida
vazoes_vertidas = vazao_vertida_pop(niveis['NIVELBARRAGEM'])

# Vazão comportas 
#vazoes_comportas = vazao_comporta_pop(abertura['comporta 1'], abertura['comporta2'])

# Somar vazoes para determinar vazão defluente total
# TODO: ajustar depois de calcular todas as vazões
vazoes = vazoes_maquinas.sum(axis=1) + vazoes_vertidas

# Remover vazões abaixo de um valor
vazoes = vazoes[vazoes > 0]
niveis = niveis.loc[vazoes.index]

# Limpar dados muito divergentes
x, y = remove_outliers(vazoes, niveis['NIVELBARRAGEM'], threshold=3.5, steps=1)
#y, x = remove_outliers(niveis['NIVELBARRAGEM'], vazoes, threshold=1.5, steps=20)
x = x.values.flatten()
y = y.values.flatten()

# --- Determinar Início do processo iterativo para determinação da curva chave, com as vazões ajustadas
fit_curva_chave = curve_fit(curva_chave, x, y)

x_curva_eq = np.arange(min(x), max(x), 0.1)
y_curva_eq = curva_chave(x_curva_eq, fit_curva_chave[0][0], fit_curva_chave[0][1], fit_curva_chave[0][2])

plt.figure()
#plt.scatter(vazoes, niveis['NIVELJUSANTE'])
plt.grid()
plt.scatter(x, y, s=10, c='#7a7a78')
plt.plot(x_curva_eq, y_curva_eq, color='k')
plt.legend(['Dados Usina', 'Curva'])
plt.xlabel('Vazão [m³/s]')
plt.ylabel('Nível Altimétrico [m]')

plt.savefig('ferramentas_hidro/curva_chave_jusante_pop/figs/fit_curva.png', bbox_inches='tight')

print("Equação da curva chave: {} * Q ^ {} + {}".format(fit_curva_chave[0][2], fit_curva_chave[0][0], fit_curva_chave[0][1]))

print('End')