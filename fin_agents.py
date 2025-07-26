import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import requests
import json
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# load api key
load_dotenv("api_key.env")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

class FinanceState(TypedDict):
    query: str
    user_profiles: str
    market_data: str
    portfolio_analysis: str
    budget_analysis: str
    risk_assessment: str
    final_recommendation: str

class FinanceMultiAgent:
    def __init__(self):
        self.setup_agents()
        self.user_profiles = {}
    
    def setup_agents(self):
        # initializing agents to multi-agent system

        #  ------------- PORTFOLIO ANALYST AGENT -------------
        @tool
        def get_stock_price(symbol: str) -> str:
            """Get current stock price and basic info"""
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                history = stock.history(period="1d")

                if history.empty:
                    return f"No data found for {symbol}"
                
                current_price = history["Close"].iloc[-1]
                prev_close = history["Close"].iloc[-2]
                change = current_price - prev_close
                change_percentage = (change / prev_close) * 100 if prev_close != 0 else 0
                
                return f"""
                Stock: {symbol.upper()}
                Company: {info.get('longName', 'N/A')}
                Current Price: ${current_price:.2f}
                Change: ${change:.2f} ({change_percentage:.2f}%)
                Market Cap: ${info.get('marketCap', 0):,}
                P/E Ratio: {info.get('trailingPE', 'N/A')}
                52-Week High: ${info.get('fiftyTwoWeekHigh', 'N/A')}
                52-Week Low: ${info.get('fiftyTwoWeekLow', 'N/A')}
                """
            
            except Exception as e:
                return f"Error getting stock data for {symbol}: {str(e)}"
            
        @tool
        def analyze_portfolio(stocks: str, amounts: str) -> str:
            """Analyze a portfolio of stocks with investment amounts"""
            try:
                stock_list = [s.strip() for s in stocks.split(",")]
                amounts_list = [float(a.strip()) for a in amounts.split(",")]

                if len(stock_list) != len(amounts_list):
                    return "Error: Number of stocks and amounts must match"
                
                portfolio_data = []
                total_value = sum(amounts_list)
                
                for symbol, amount in zip(stock_list, amounts_list):
                    stock = yf.Ticker(symbol.upper())
                    info = stock.info

                    portfolio_data.append({
                        "stock": symbol.upper(),
                        "amount": amount,
                        "weight": (amount / total_value) * 100,
                        "sector": info.get("sector", "N/A"),
                        "beta": info.get("beta", "N/A"),
                    })

                # Calculate diversification
                sectors = {}

                for item in portfolio_data:
                    sector = item["sector"]
                    sectors[sector] = sectors.get(sector, 0) + item["weight"]

                analysis = f"Portfolio Analysis (Total: ${total_value:.2f})\n\n"
                for item in portfolio_data:
                    analysis += f"{item['stock']}: ${item['amount']:,.2f} ({item['weight']:.1f}%) - {item['sector']}\n"

                analysis += f"\nSector Distribution:\n"
                for sector, weight in sectors.items():
                    analysis += f"{sector}: {weight:.1f}%\n"
                
                # risk assessment
                if len(set(item["sector"] for item in portfolio_data if item["sector"] != "N/A")) < 3:
                    analysis += "\n WARNING: Portfolio lacks sector diversification\n"
                
                return analysis
            
            except Exception as e:
                return f"Portfolio analysis error: {str(e)}"
        
        #  ------------- BUDGET PLANNING AGENT -------------
        @tool
        def analyze_budget(income: float, expenses: str) -> str:
            """Analyze monthly budget and provide recommendations"""
            try: 
                expense_dict = {}
                for item in expenses.split(","):
                    category, amount = item.split(":")
                    expense_dict[category.strip()] = float(amount.strip())

                total_expenses = sum(expense_dict.values())
                savings = income - total_expenses
                savings_rate = (savings / income) * 100 if income > 0 else 0
                
                analysis = f"""
                Budget Analysis:
                Monthly Income: ${income:,.2f}
                Total Expenses: ${total_expenses:,.2f}
                Net Savings: ${savings:,.2f}
                Savings Rate: {savings_rate:.1f}%

                Expense Breakdown:
                """

                for category, amount in expense_dict.items():
                    percentage = (amount / income) * 100 if income > 0 else 0
                    analysis += f"{category.title()}: ${amount:,.2f} ({percentage:.1f}%)\n"
                
                # recommendations
                analysis += "\nRecommendations:\n"
                if savings_rate < 10:
                    analysis += "⚠️ Low savings rate. Aim for at least 10-20% of income.\n"
                elif savings_rate < 20:
                    analysis += "✅ Good savings rate. Consider increasing to 20%+.\n"
                else:
                    analysis += "🎉 Excellent savings rate!\n"
                
                # 50/30/20 rule
                needs_pct = sum(expense_dict.get(cat, 0) for cat in ['rent', 'utilities', 'groceries', 'transport']) / income * 100
                if needs_pct > 50:
                    analysis += "💡 Consider reducing essential expenses (currently {:.1f}%, target: 50%)\n".format(needs_pct)
                
                return analysis
            
            except Exception as e:
                return f"Budget analysis error: {str(e)}"
        
        @tool
        def suggest_budget_improvements(current_budget: str) -> str:
            """Suggest specific budget improvements"""
            suggestions = """
            Budget Improvement Strategies:
            
            1. Reduce Fixed Costs:
                • Negotiate rent or consider moving
                • Review subscription services
                • Shop for better insurance rates
            
            2. Optimize Variable Expenses:
                • Meal planning and cooking at home
                • Use public transportation
                • Find free entertainment options
            
            3. Increase Income:
                • Side hustles or freelancing
                • Skill development for promotions
                • Passive income streams
            
            4. Automate Savings:
                • Set up automatic transfers
                • Use high-yield savings accounts
                • Consider investment accounts
            """

            return suggestions
        
        #  ------------- MARKET RESEARCH AGENT -------------
        @tool
        def get_market_overview() -> str:
            """Get current market overview and trends"""
            try:
                # major indices
                indices = {
                    "^GSPC": "S&P 500",
                    "^DJI": "Dow Jones",
                    "^IXIC": "NASDAQ",
                }

                market_data = "Market Overview:\n\n"
                for symbol, name in indices.items():
                    try:
                        ticker = yf.Ticker(symbol)
                        history = ticker.history(period="2d")
                        if not history.empty:
                            current = history["Close"].iloc[-1]
                            previous = history["Close"].iloc[-2] if len(history) > 1 else current
                            change = current - previous
                            change_percentage = (change / previous) * 100 if previous != 0 else 0

                            market_data += f"{name}: {current:.2f} ({change:+.2f}, {change_percentage:+.2f}%)\n"
                    except:
                        market_data += f"{name}: Data unavailable\n"
                
                return market_data
            
            except Exception as e:
                return f"Market data error: {str(e)}"
        
        @tool
        def get_economic_indicators() -> str:
            """Get economic indicators and news"""
            return """
            Key Economic Factors to Watch:
            
            📊 Interest Rates: Federal Reserve policy affects borrowing costs
            📈 Inflation: Impacts purchasing power and investment returns
            💼 Employment: Job market health indicates economic strength
            🏠 Housing Market: Real estate trends affect wealth
            💱 Currency Strength: USD performance impacts international investments
            
            💡 Tip: Check Federal Reserve announcements and economic calendars
            for the latest updates on these indicators.
            """
        
        #  ------------- RISK ASSESSMENT AGENT -------------
        @tool
        def assess_investment_risk(age: int, income: float, risk_tolerance: str) -> str:
            """Access appropriate investment risk level"""
            try:
                age_factor = (100 - age) / 100
                
                risk_scores = {"low": 1, "medium": 2, "high": 3}
                tolerance_score = risk_scores.get(risk_tolerance.lower(), 2)

                # risk recommendation
                if age < 30 and tolerance_score >= 2:
                    risk_level = "Aggressive"
                    equity_allocation = "80-90%"
                elif age < 50 and tolerance_score >= 2:
                    risk_level = "Moderate to Aggressive"
                    equity_allocation = "60-80%"
                elif age < 65:
                    risk_level = "Moderate"
                    equity_allocation = "40-60%"
                else:
                    risk_level = "Conservative"
                    equity_allocation = "20-40%"
                
                assessment = f"""
                Age: {age}
                Risk Tolerance: {risk_tolerance.title()}
                Recommended Risk Level: {risk_level}
                Suggested Equity Allocation: {equity_allocation}
                Bond Allocation: {100 - int(equity_allocation.split('-')[0][:-1])}%+
                
                Key Considerations:
                • Time horizon until retirement: {65 - age} years
                • Emergency fund: 3-6 months of expenses recommended
                • Diversification across asset classes and geographies
                """
                
                if age < 35:
                    assessment += "\n💡 Young investor advantage: Long time horizon allows for higher risk/return potential"
                elif age > 55:
                    assessment += "\n💡 Pre-retirement focus: Consider gradually shifting to more conservative allocations"
                
                return assessment
            
            except Exception as e:
                return f"Risk assessment error: {str(e)}"
            
        @tool
        def calculate_emergency_fund(monthly_expenses: float, job_stability: str) -> str:
            """Calculate recommended emergency fund size"""
            try:
                stability_multipliers = {
                    'stable': 3,
                    'moderate': 6,
                    'unstable': 9
                }
                
                multiplier = stability_multipliers.get(job_stability.lower(), 6)
                recommended_fund = monthly_expenses * multiplier
                
                return f"""
                Emergency Fund Recommendation:
                
                Monthly Expenses: ${monthly_expenses:,.2f}
                Job Stability: {job_stability.title()}
                
                Recommended Emergency Fund: ${recommended_fund:,.2f}
                ({multiplier} months of expenses)
                
                💡 Tips:
                • Keep in high-yield savings account
                • Separate from investment accounts
                • Review and adjust annually
                • Consider money market funds for larger amounts
                """
                
            except Exception as e:
                return f"Emergency fund calculation error: {str(e)}"
            
        # Creating specialized agents
        self.portfolio_agent = create_react_agent(
            llm,
            tools=[get_stock_price, analyze_portfolio],
            prompt="You are a portfolio analysis specialist. Help users analyze stocks and optimize their investment portfolios.",
        )

        self.budget_agent = create_react_agent(
            llm,
            tools=[analyze_budget, suggest_budget_improvements],
            prompt="You are a budget planning specialist. Help users create and optimize their personal budgets.",
        )

        self.market_research_agent = create_react_agent(
            llm,
            tools=[get_market_overview, get_economic_indicators],
            prompt="You are a market research specialist. Provide market insights and economic analysis"
        )

        self.risk_assessment_agent = create_react_agent(
            llm,
            tools=[assess_investment_risk, calculate_emergency_fund],
            prompt="You are a risk assessment specialist. Evaluate financial risks and provide safety recommendations.",
        )
    
    # creting nodes for langgraph
    def portfolio_analysis_node(self, state: FinanceState) -> FinanceState:
        """Portfolio analysis specialist"""
        print("📊 Portfolio Analyst working...")
        config = {"configurable": {"thread_id": "portfolio_agent"}}
        response = self.portfolio_agent.invoke(
            {"messages": [HumanMessage(content=f"Analyze this request: {state['query']}")]},
            config
        )
        
        state["portfolio_analysis"] = response["messages"][-1].content
        return state
    
    def budget_analysis_node(self, state: FinanceState) -> FinanceState:
        """Budget planning specialist"""
        print("💰 Budget Planner working...")
        
        config = {"configurable": {"thread_id": "budget_agent"}}
        response = self.budget_agent.invoke(
            {"messages": [HumanMessage(content=f"Analyze this request: {state['query']}")]},
            config
        )
        
        state["budget_analysis"] = response["messages"][-1].content
        return state
    
    def market_research_node(self, state: FinanceState) -> FinanceState:
        """Market research specialist"""
        print("📈 Market Researcher working...")
        
        config = {"configurable": {"thread_id": "market_agent"}}
        response = self.market_research_agent.invoke(
            {"messages": [HumanMessage(content=f"Research this request: {state['query']}")]},
            config
        )
        
        state["market_data"] = response["messages"][-1].content
        return state
    
    def risk_assessment_node(self, state: FinanceState) -> FinanceState:
        """Risk assessment specialist"""
        print("⚖️ Risk Assessor working...")
        
        config = {"configurable": {"thread_id": "risk_agent"}}
        response = self.risk_assessment_agent.invoke(
            {"messages": [HumanMessage(content=f"Assess risks for: {state['query']}")]},
            config
        )
        
        state["risk_assessment"] = response["messages"][-1].content
        return state
    
    def coordinator_node(self, state: FinanceState) -> FinanceState:
        """Coordinate all analyses into final recommendation"""
        print("🎯 Financial Coordinator synthesizing recommendations...")


        prompt = f"""
        As a senior financial advisor, synthesize the following analyses into a comprehensive recommendation:
        
        Original Query: {state['query']}
        
        Portfolio Analysis: {state.get('portfolio_analysis', 'N/A')}
        Budget Analysis: {state.get('budget_analysis', 'N/A')}
        Market Research: {state.get('market_data', 'N/A')}
        Risk Assessment: {state.get('risk_assessment', 'N/A')}
        
        Provide a clear, actionable financial recommendation with:
        1. Summary of key findings
        2. Specific action steps
        3. Risk considerations
        4. Timeline for implementation
        """

        response = llm.invoke([HumanMessage(content=prompt)])
        state["final_recommendation"] = response.content
        return state
    
    def create_workflow(self):
        """Create the financial advisory workflow"""
        workflow = StateGraph(FinanceState)

        # adding all specialist nodes
        workflow.add_node("portfolio_analysis", self.portfolio_analysis_node)
        workflow.add_node("budget_analysis", self.budget_analysis_node)
        workflow.add_node("market_research", self.market_research_node)
        workflow.add_node("risk_assessment", self.risk_assessment_node)
        workflow.add_node("coordinator", self.coordinator_node)

        # parallel execution of specialists
        workflow.set_entry_point("portfolio_analysis")
        workflow.add_edge("portfolio_analysis", "budget_analysis")
        workflow.add_edge("budget_analysis", "market_research")
        workflow.add_edge("market_research", "risk_assessment")
        workflow.add_edge("risk_assessment", "coordinator")
        workflow.add_edge("coordinator", END)

        return workflow.compile()
    
    def get_financial_advice(self, query: str) -> str:
        """Main function to get comprehensive financial advice"""
        print(f"🏦 Financial Advisory Team analyzing: {query}")
        print("=" * 60)
        
        app = self.create_workflow()
        
        initial_state = {
            "query": query,
            "user_profiles": "",
            "market_data": "",
            "portfolio_analysis": "",
            "budget_analysis": "",
            "risk_assessment": "",
            "final_recommendation": ""
        }
        
        final_state = app.invoke(initial_state)
        return final_state["final_recommendation"]
    
def main():
    """Interactive financial advisory interface"""
    print("🏦 Personal Finance Multi-Agent Advisory System")
    print("=" * 60)
    print("Meet your financial advisory team:")
    print("📊 Portfolio Analyst - Stock analysis & portfolio optimization")
    print("💰 Budget Planner - Personal budgeting & expense tracking") 
    print("📈 Market Researcher - Economic trends & market analysis")
    print("⚖️ Risk Assessor - Investment risk & safety planning")
    print("🎯 Financial Coordinator - Synthesizes all recommendations")
    print("=" * 60)
    
    system = FinanceMultiAgent()
    
    print("\nExample queries:")
    print("• 'Analyze my portfolio: AAPL,GOOGL,TSLA with amounts 5000,3000,2000'")
    print("• 'Review my budget: income 6000, expenses rent:2000,food:800,transport:400'")
    print("• 'I'm 25 years old, income 70k, medium risk tolerance - what should I invest in?'")
    print("• 'What's the current market outlook?'")
    print("\nType 'quit' to exit")
    print("-" * 60)
    
    while True:
        try:
            query = input("\n💬 Financial Question: ").strip()
            
            if query.lower() == 'quit':
                print("👋 Thank you for using the Financial Advisory System!")
                break
            
            if not query:
                print("Please enter a financial question.")
                continue
            
            advice = system.get_financial_advice(query)
            print(f"\n📋 Answer:\n{advice}")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()