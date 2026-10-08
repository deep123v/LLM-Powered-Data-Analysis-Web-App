import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings

st.set_page_config(
    page_title="LLM Powered Data Analysis App",
    page_icon="🤖",
    layout="wide"
)

# Initialise Groq Client using OpenAI compatibility
client = openai.OpenAI(
    api_key=st.secrets['GROQ_API_KEY'],
    base_url="https://api.groq.com/openai/v1"
)


# Session state Initialization
if "message" not in st.session_state:
    st.session_state.message = []
if "df" not in st.session_state:
    st.session_state.df = None
if "data_summary" not in st.session_state:
    st.session_state.data_summary = None

st.title("LLM Powered Data Analysis App")
st.markdown("upload data and ask questions in plain english!")

# Sidebar for file upload
with st.sidebar:
    st.header("Data Upload")
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"]
    )

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        
        # Populate data summary for token optimization / context
        st.session_state.data_summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'sample': df.head(3).to_string(),
            'stats': df.describe().to_string()
        }
        
        st.success(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns")

        # Data Preview
        with st.expander("Data Preview"):
            st.dataframe(df.head())

        # basic statistics
        with st.expander("Basic Statistics"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Number of Rows", df.shape[0])
                st.metric("Number of Columns", df.shape[1])
            with col2:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
                st.metric("Number of Missing Values", df.isnull().sum().sum())

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.info("Please upload a valid CSV file format.")

else:
    st.info("Please upload a CSV file to start analyzing.")

# Main Chat interface
if st.session_state.df is not None:
    # Display chat history
    for msg in st.session_state.message:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Re-Display any saved figures
            if 'figure' in msg:
                st.pyplot(msg['figure'])

    # Chat input
    user_input = st.chat_input("Ask a question about your data...")
    if user_input:
        # Add user message to history
        st.session_state.message.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Prepare data context with token optimisation
        df = st.session_state.df

        if len(df) > 100:
            data_context = f"""
Dataset shape: {st.session_state.data_summary['shape']}
Columns: {', '.join(str(c) for c in st.session_state.data_summary['columns'])}
Data types: {st.session_state.data_summary['dtypes']}
Sample rows: {st.session_state.data_summary['sample']}
Basic statistics: {st.session_state.data_summary['stats']}
"""
        else:
            data_context = f"""
Full dataset:
{df.to_string()}
"""

        # system prompt
        system_prompt = f"""You are a helpful data analyst assistant.
The user has uploaded a CSV file with the following information: {data_context}

The data is loaded in a pandas DataFrame called 'df'.

Guidelines:
1. Answer the user's question clearly and concisely
2. If the question required analysis, write python code using pandas, matplotlib, or seaborn
3. For visualizations, always use plt.figure() before plotting and include plt.tight_layout()
4. Always validate data before operations (Check for nulls, data type etc.)
5. If you can't answer due to data limitations, explain why
6. Keep responses focused on the data and questions asked

When writing codes:
1. Import statements are already done (pandas as pd, matplotlib.pyplot as plt, seaborn as sns)
2. The dataframe is available as 'df'
3. For plots, use plt.figure(figsize=(10,6)) for better display
4. Always add titles and labels to plots
"""

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Generating response..."):
                try:
                    # Get conversation history context
                    messages = [
                        {"role": "system", "content": system_prompt}
                    ]

                    # include last 3 exchange for context
                    for msg in st.session_state.message[-3:]:
                        # Truncate long message in history to save tokens
                        content = msg["content"]
                        if len(content) > 500:
                            content = content[:500] + "..."
                        messages.append({"role": msg["role"], "content": content})

                    messages.append({"role": "user", "content": user_input})

                    response = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=messages,
                        temperature=0.1,
                        max_tokens=1500
                    )
                    reply = response.choices[0].message.content
                    st.markdown(reply)

                    # try to execute any codes in the response
                    if "```python" in reply:
                        code_blocks = reply.split("```python")
                        for i in range(1, len(code_blocks)):
                            code = code_blocks[i].split("```")[0]

                            try:
                                # Capture warnings
                                with warnings.catch_warnings(record=True) as w:
                                    warnings.simplefilter('always')

                                # create figure for potential plots
                                plt.figure(figsize=(10, 6))

                                # Execute code in controlled environment
                                exec_globals = {
                                    'df': df,
                                    'pd': pd,
                                    'plt': plt,
                                    'sns': sns,
                                    'st': st
                                }
                                exec(code.strip(), exec_globals)

                                # Display any captured warnings
                                if w:
                                    for warning in w:
                                        st.info(f"Note: {warning.message}")

                                # Display plot if created
                                fig = plt.gcf()
                                if fig.get_axes():
                                    st.pyplot(fig)
                                    # Save figure in message for persistence
                                    st.session_state.message.append(
                                        {
                                            "role": "assistant",
                                            "content": reply,
                                            "figure": fig
                                        }
                                    )
                                else:
                                    st.session_state.message.append({
                                        'role': 'assistant',
                                        'content': reply
                                    })
                                plt.close()

                            except Exception as e:
                                error_type = type(e).__name__
                                st.error(f"Code execution failed: {error_type}")

                                # Provide helpful context based on error type
                                if "NameError" in str(e):
                                    st.info("This might mean a column name is misspelled or doesn't exist.")
                                elif "TypeError" in str(e):
                                    st.info("This often happens when trying to plot non-numeric data.")
                                elif "KeyError" in str(e):
                                    st.info("The specified column might not exist in your dataset.")
                                else:
                                    st.info("Try rephrasing your question or check your data format.")

                                st.code(code, language="python")

                    # Save assistant response to chat history if no code execution happened or already handled
                    if "```python" not in reply:
                        st.session_state.message.append({"role": "assistant", "content": reply})

                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.info("Please check your Groq API key and try again.")

else:
    # No data uploaded state
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("Please upload a CSV file to start analyzing.")

        # Example questions
        st.markdown("### 💡 Example questions you can ask:")

        st.markdown("""
        - What are the main trends in my data?
        - Show me a correlation matrix
        - Create a bar chart of the top 10 categories
        - What's the average value by month?
        - Are there any outliers in the price column?
        """)

# footer with tips
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
💡 Tip: be specific with your question for better results |
🔒 Your data stays private and is not stored
</div>
""", unsafe_allow_html=True)