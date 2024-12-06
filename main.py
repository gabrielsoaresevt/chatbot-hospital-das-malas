from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import numpy as np
import pandas as pd
import google.generativeai as genai

app = Flask(__name__)

@app.route('/bot', methods=['POST'])
def bot():
    user_message = request.values.get('Body', '')

    GOOGLE_API_KEY = "<CHAVE_API>"
    genai.configure(api_key=GOOGLE_API_KEY)

    DOCUMENT1 = {
        "Titulo": "Saudações iniciais e boas-vindas ao Hospital das Malas",
        "Conteudo": "Olá, tudo bem? Somos o Hospital das Malas, especialistas em consertos e restaurações de bolsas, malas e mais.\n\nComo podemos te ajudar? (Este documento é para ser usado quando o cliente enviar mensagens de saudação como 'Oi', 'Olá', ou similares.)"
    }

    DOCUMENT2 = {
        "Titulo": "Horários de funcionamento e atendimento aos finais de semana, localização e onde trabalhamos",
        "Conteudo": "Estamos funcionando de segunda a sexta-feira das 8h até às 20h. \n\nPara atendimento aos finais de semana e feriados, é necessário combinar previamente.\n\nVocê pode nos encontrar na R. Dinieper, 172 - Vila Ipojuca, localizado em frente à Praça Regência. Estamos te esperando!"
    }

    DOCUMENT3 = {
        "Titulo": "Lista de serviços oferecidos no Hospital das Malas e com o que trabalhamos",
        "Conteudo": "No Hospital das Malas oferecemos os seguintes serviços:\n\n- Consertos de malas e bolsas;\n- Troca de zíperes, fechos e alças;\n- Reforço de costuras;\n- Manutenção de mochilas e acessórios de viagem."
    }

    DOCUMENT4 = {
        "Titulo": "Nossa especialidade em consertos e restaurações",
        "Conteudo": "Somos especialistas em consertos e restaurações utilizando técnicas precisas e materiais de qualidade. Garantimos o melhor resultado para prolongar a durabilidade e manter a elegância dos seus acessórios."
    }

    DOCUMENT5 = {
        "Titulo": "Quais valores, custos, orçamentos para um serviço, trabalho ou conserto",
        "Conteudo": "Para saber mais sobre orçamentos ou o custo de um serviço, por favor aguarde um momento até que um de nossos atendentes humanos possa ajudá-lo."
    }

    DOCUMENT6 = {
        "Titulo": "Confiança, qualidade e atendimento personalizado",
        "Conteudo": "Com anos de experiência, nossa equipe é composta por profissionais altamente qualificados, prontos para oferecer atendimento personalizado e eficiente."
    }
    
    DOCUMENT7 = {
        "Titulo": "Prazos e tempo de entrega de um serviço/conserto/trabalho",
        "Conteudo": "Os prazos dependem do serviço/trabalho que será realizado, geralmente de 1 a 2 semanas"
    }
    
    DOCUMENT8 = {
        "Titulo": "Quero falar com um humano, dono do estabelecimento, ou preciso de uma ajuda específica",
        "Conteudo": "Aguarde um momento, até que um humano possa atende-lo"
    }
    
    DOCUMENT9 = {
        "Titulo": "Quero arrumar minha mala ou bolsa, ou mochila ou solicitar algum serviço",
        "Conteudo": "Aguarde um momento, até que um humano possa atende-lo"
    }
    
    DOCUMENT_DEFAULT = {
        "Titulo": "Resposta genérica para mensagens não reconhecidas",
        "Conteudo": "Desculpe, não consegui encontrar informações sobre isso. Por favor, tente reformular sua pergunta ou entre em contato com um atendente humano para ajudar melhor."
    }

    documents = [DOCUMENT1, DOCUMENT2 , DOCUMENT3, DOCUMENT4, DOCUMENT5, DOCUMENT6, DOCUMENT7, DOCUMENT8, DOCUMENT9, DOCUMENT_DEFAULT]

    ### Gerando dataframe

    df = pd.DataFrame(documents)
    df.columns = ["Titulo", "Conteudo"]
    df.head()

    ### Criando função de incorporação

    model =  "models/embedding-001"

    def embed_fn(title, text):
      return genai.embed_content(
        model=model,
        content=text,
        title=title,
        task_type="RETRIEVAL_DOCUMENT"
        )["embedding"]

    ### Criar e popular columa Embeddings

    df["Embeddings"] = df.apply(lambda row: embed_fn(row["Titulo"], row["Conteudo"]), axis=1)
    df.head()

    ### Criando a função de gerar embedding da consulta e buscar nos documentos

    def create_and_find(consulta, base, model):
      embedding_da_consulta = genai.embed_content(
          model=model,
          content=consulta,
          task_type="RETRIEVAL_QUERY"
          )["embedding"]

      dot_products = np.dot(np.stack(df["Embeddings"]), embedding_da_consulta)

      index = np.argmax(dot_products)
      return df.iloc[index]["Conteudo"]

    ### Consultar informações dos documentos 

    consultation = user_message
    text = create_and_find(consultation, df, model)

    generation_config = {
        "temperature": 0,
        "candidate_count": 1
    }

    gen_model = genai.GenerativeModel("gemini-1.0-pro",
                                generation_config=generation_config)

    ### Gerando texto a partir do resultado da consulta

    prompt_base = "Reescreva esse texto de uma forma, sem adicionar informações que não façam parte do texto e sem utilizar gírias"
    chatbot_response = gen_model.generate_content((f"{prompt_base} {text}"))
    
    response = MessagingResponse()
    message = response.message()
    
    message.body(chatbot_response.text)
     
    return str(response)
    
@app.route('/')
def index():
    return "Rout test"
  
if __name__ == '__main__':
    app.run(host='0.0.0.0')