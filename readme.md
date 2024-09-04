**Em andamento**


# Detecção de Radiografias Odontológicas com Detectron2 e GPT

Este projeto utiliza o Detectron2 para detectar e classificar áreas de interesse em radiografias odontológicas e o GPT da OpenAI para gerar pré-laudos automáticos.

## Visão Geral

1. **Configuração do Ambiente**: Importação de bibliotecas e configuração da API do GPT.
2. **Processamento de Dados**: Carregamento e mapeamento de categorias a partir do JSON de anotações.
3. **Configuração do Modelo**: Registro do dataset e configuração do Detectron2 com pesos treinados.
4. **Inferência**: Predição em radiografias, extração de classes e probabilidades.
5. **Geração de Laudo**: Envio das predições e imagem para o GPT, que gera um pré-diagnóstico.
6. **Visualização**: Exibição das imagens e do pré-laudo gerado.

## Exemplo de Uso

**Texto do Pré-Laudo:**
```
**Pré-Laudo de Radiografia Odontológica**

**Paciente:** Nome do Paciente  
**Data da Coleta:** Data do Exame  
**Exame Realizado:** Radiografia Odontológica (Tipo)  
**Método de Análise:** Análise por modelo de Detecção de Objetos (Detectron2)

---

**Descrição do Exame:**

A análise foi realizada com base na imagem fornecida, gerando as seguintes classes preditas e suas respectivas probabilidades:

1. **Crown** - Probabilidade: 82.19%
   - **Observação:** Indica a presença de uma coroa dentária, que pode estar associada a um tratamento restaurador ou protético.

2. **Filling** - Probabilidade: 72.71%
   - **Observação:** Sugere a presença de material de restauração, indicando que o dente pode ter sido submetido a um tratamento por cárie.

3. **Impacted Tooth** - Probabilidade: 66.61%
   - **Observação:** Indica a possibilidade de um dente impactado, que pode necessitar de avaliação adicional e, possivelmente, intervenção cirúrgica.

4. **Root Canal Treatment** - Probabilidade: 61.59%
   - **Observação:** Sugere que o dente pode ter passado por tratamento de canal, uma intervenção necessária em casos de inflamação ou infecção da polpa dental.

---

**Interpretação:**

Com base nas classes identificadas, é possível concluir que:

- O paciente apresenta ao menos um dente com uma coroa e uma ou mais restaurações. A presença de um dente impactado e a realização de um tratamento de canal também são notáveis e requerem consideração clínica.

---

**Recomendações:**

1. **Avaliação Clínica:** É recomendada a consulta com um dentista para discutir os achados e considerar a necessidade de um exame clínico mais detalhado.
  
2. **Tratamentos Adicionais:** Dependendo do estado clínico e dos sintomas do paciente, poderá ser necessário intervenções adicionais, como extração do dente impactado ou revisão do tratamento de canal.

3. **Acompanhamento:** Sugere-se um retorno para controle após finalização de qualquer tratamento.

---

**Assinatura:**
[Seu Nome]  
[Seu Cargo]  
[Instituição de Saúde]  
[Data]  

---

Este laudo é um resumo da análise realizada pela ferramenta de detecção e não substitui a avaliação clínica profissional.
```


**Imagem com Detecções:**

![output](https://github.com/user-attachments/assets/04927dd7-9254-465b-9b78-ca3fad212f93)

## Link para o Notebook

Confira o notebook completo do treinamento [https://www.kaggle.com/code/henriquerezermosqur/mask-r-cnn](#).
