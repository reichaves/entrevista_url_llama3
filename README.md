# Chatbot de Websites com Llama 3 🤖

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://entrevista-sites-llama3.streamlit.app/)
[![Open in Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/reichaves/chatbot-websites-llama-3.2-90b-text-preview-Brazil)

Este projeto implementa um sistema de Retrieval-Augmented Generation (RAG) conversacional para entrevistar o conteúdo de URLs, utilizando Streamlit, LangChain e modelos de linguagem de grande escala. Agradeço às aulas de [Krish C Naik](https://www.youtube.com/user/krishnaik06)

## Funcionalidades

- Processamento e análise do conteúdo de websites específicos
- Geração de respostas usando o modelo llama-3.2-90b-text-preview da Meta
- Embeddings de texto usando o modelo all-MiniLM-L6-v2 do Hugging Face
- Interface de chat interativa para perguntas e respostas
- Suporte para múltiplos idiomas (com foco em Português do Brasil)

## Como usar

1. Acesse o aplicativo através do Streamlit ou Hugging Face Spaces (links acima).
2. Insira suas chaves de API para Groq e Hugging Face.
3. Digite a URL do website que deseja analisar.
4. Faça perguntas sobre o conteúdo do website no chat.
5. O chatbot responderá com informações baseadas no conteúdo processado.

## Requisitos

- Chave de API Groq
- Token de API Hugging Face (com permissões de escrita)

## Tecnologias utilizadas

- Python
- Streamlit
- LangChain
- Groq (Llama 3.2-90b-text-preview)
- Hugging Face Embeddings (all-MiniLM-L6-v2)
- BeautifulSoup
- FAISS

## Configuração local

Para executar este projeto localmente:

1. Clone o repositório
2. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```
3. Configure as variáveis de ambiente para GROQ_API_KEY e HUGGINGFACEHUB_API_TOKEN
4. Execute o aplicativo:
   ```
   streamlit run app.py
   ```

## Considerações éticas

- Evite compartilhar URLs com dados sensíveis, pessoais ou de propriedade intelectual.
- O conteúdo processado pode ser usado para treinar o modelo de IA.
- Verifique sempre as informações geradas com as fontes originais.

## Contribuições

Contribuições são bem-vindas! Por favor, abra uma issue ou pull request para sugestões de melhorias.

## Autor

Desenvolvido por Reinaldo Chaves (reichaves@gmail.com)

## Licença

Este projeto está sob a licença [inserir tipo de licença aqui].
