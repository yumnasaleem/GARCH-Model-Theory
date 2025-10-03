import tkinter as tk
from tkinter import ttk, messagebox
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from arch import arch_model

class GARCHModelApp:
    def __init__(self, master):
        self.master = master
        master.title("GARCH Model GUI")
        master.geometry("1000x800") # Set initial window size

        self.ticker_label = tk.Label(master, text="Ticker Symbol (e.g., SPY):")
        self.ticker_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.ticker_entry = tk.Entry(master)
        self.ticker_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.ticker_entry.insert(0, "SPY") # Default value

        self.start_date_label = tk.Label(master, text="Start Date (YYYY-MM-DD):")
        self.start_date_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.start_date_entry = tk.Entry(master)
        self.start_date_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.start_date_entry.insert(0, "2010-01-01")

        self.end_date_label = tk.Label(master, text="End Date (YYYY-MM-DD):")
        self.end_date_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.end_date_entry = tk.Entry(master)
        self.end_date_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        self.end_date_entry.insert(0, "2023-12-31")

        self.p_label = tk.Label(master, text="GARCH Order p:")
        self.p_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.p_entry = tk.Entry(master)
        self.p_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        self.p_entry.insert(0, "1") # Default GARCH(1,1)

        self.q_label = tk.Label(master, text="GARCH Order q:")
        self.q_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.q_entry = tk.Entry(master)
        self.q_entry.grid(row=4, column=1, padx=5, pady=5, sticky="ew")
        self.q_entry.insert(0, "1") # Default GARCH(1,1)

        self.run_button = tk.Button(master, text="Run GARCH Model", command=self.run_garch_model)
        self.run_button.grid(row=5, column=0, columnspan=2, pady=10)

        # Output Text Box for Summary
        self.output_label = tk.Label(master, text="Model Summary:")
        self.output_label.grid(row=6, column=0, padx=5, pady=5, sticky="w")
        self.output_text = tk.Text(master, height=15, width=60, wrap="word")
        self.output_text.grid(row=7, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        # Matplotlib Figure and Canvas
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=2, rowspan=8, padx=10, pady=5, sticky="nsew")

        # Toolbar for matplotlib plot
        self.toolbar_frame = tk.Frame(master)
        self.toolbar_frame.grid(row=8, column=2, padx=10, pady=5, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        # Configure grid weights for resizing
        master.grid_columnconfigure(1, weight=1) # Allow entry fields to expand
        master.grid_columnconfigure(2, weight=3) # Allow plot area to expand
        master.grid_rowconfigure(7, weight=1) # Allow output text to expand

    def run_garch_model(self):
        ticker = self.ticker_entry.get().upper()
        start_date = self.start_date_entry.get()
        end_date = self.end_date_entry.get()
        try:
            p = int(self.p_entry.get())
            q = int(self.q_entry.get())
            if p < 0 or q < 0:
                raise ValueError("p and q must be non-negative integers.")
        except ValueError:
            messagebox.showerror("Input Error", "GARCH orders p and q must be non-negative integers.")
            return

        self.output_text.delete(1.0, tk.END) # Clear previous output
        self.output_text.insert(tk.END, "Fetching data...\n")
        self.master.update_idletasks() # Update GUI immediately

        try:
            # Fetch data
            data = yf.download(ticker, start=start_date, end=end_date)['Close']
            if data.empty:
                messagebox.showerror("Data Error", f"No data found for {ticker} in the specified date range.")
                return

            # Calculate daily returns (percentage change for 'arch' library)
            returns = 100 * data.pct_change().dropna() # Convert to percentage for better scaling in arch_model
            if returns.empty:
                messagebox.showerror("Data Error", "Not enough data to calculate returns.")
                return

            self.output_text.insert(tk.END, f"Data fetched for {ticker}. Running GARCH({p},{q}) model...\n")
            self.master.update_idletasks()

            # Fit GARCH model
            # mean='Zero' if you assume returns have a zero mean, or 'Constant' for a non-zero mean
            # vol='GARCH' is the default, but explicitly stated for clarity
            # p and q are the orders of the GARCH model
            model = arch_model(returns, mean='Constant', vol='GARCH', p=p, q=q)
            results = model.fit(update_freq=5, disp='off') # disp='off' suppresses verbose output during fitting

            # Display summary
            self.output_text.insert(tk.END, results.summary().as_text())
            self.output_text.insert(tk.END, "\n\nConditional Volatility (Annualized):\n")

            # Extract conditional volatility and annualize
            conditional_volatility = np.sqrt(results.conditional_volatility * (252**0.5)) # Daily volatility to annual
            
            # Update output text with annualized conditional volatility
            self.output_text.insert(tk.END, conditional_volatility.to_string())

            # Plot results
            self.ax.clear()
            self.ax.plot(returns.index, returns, label='Returns (%)', alpha=0.7)
            self.ax.plot(conditional_volatility.index, conditional_volatility, color='red', label='Annualized Conditional Volatility (%)')
            self.ax.set_title(f'{ticker} Returns and GARCH({p},{q}) Annualized Conditional Volatility')
            self.ax.set_xlabel('Date')
            self.ax.set_ylabel('Value (%)')
            self.ax.legend()
            self.ax.grid(True, linestyle='--', alpha=0.7)
            self.fig.autofmt_xdate() # Format x-axis dates nicely
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            self.output_text.insert(tk.END, f"\nError: {e}\n")

# Main part of the script
if __name__ == "__main__":
    root = tk.Tk()
    app = GARCHModelApp(root)
    root.mainloop()