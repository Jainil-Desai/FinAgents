# 💰 FinAgents: A Multi-Agent Financial Advisor

FinAgents is a beginner-friendly AI-powered personal finance advisor that simulates a team of expert agents working together to help you manage your money.

Whether you're interested in AI, finance, or both — this project offers a fun and practical way to explore concepts like multi-agent systems, budgeting, portfolio analysis, and more.

---

## 🧠 Agent Team Structure

### 1. 📊 Portfolio Analyst Agent
- Analyzes stock performance using real-time prices
- Calculates portfolio metrics (risk, return, diversification)
- Compares stock options for better investment decisions

### 2. 💸 Budget Planning Agent
- Tracks income and expenses
- Identifies spending patterns
- Suggests budget adjustments and cost-cutting measures

### 3. 🌍 Market Research Agent
- Pulls economic data and market news
- Analyzes trends in sectors and industries
- Interprets economic indicators (inflation, interest rates)

### 4. ⚠️ Risk Assessment Agent
- Evaluates portfolio volatility and risk
- Recommends emergency fund size
- Reviews potential insurance needs

### 5. 🧭 Financial Planning Agent (Coordinator)
- Coordinates across all agents
- Provides long-term strategy recommendations
- Handles retirement planning, tax efficiency, and financial goals

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/FinAgents.git
```

### 2. Navigate into the folder
```bash
cd FinAgents
```

### 3. Set up a virtual environment
```bash
python -m venv myenv
source myenv/bin/activate
```

### 4. Install the required libraries
```bash
pip install langgraph langchain-openai langchain-community python-dotenv requests beautifulsoup4 yfinance pandas numpy matplotlib
```

### 5. Add OpenAI API key

Create a file called `api_key.env` and add this line (replace with your actual key):
```bash
OPENAI_API_KEY=sk-...
```

### 6. Run the application
```bash
python fin_agents.py
```