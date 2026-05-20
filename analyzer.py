import os
import pandas as pd

class MetabolicModelAnalyzer:
    """
    A professional tool for parsing and analyzing Genome-Scale Metabolic Models (GEMs).
    Designed for handling Excel-based metabolic reconstructions.
    """
    
    def __init__(self, file_path: str):
        """Initializes the analyzer with the path to the metabolic model."""
        self.file_path = file_path
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Loads the Excel file into a Pandas DataFrame with error handling."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"❌ Target file not found at: {self.file_path}")
            
        print(f"⏳ Loading metabolic model from: {os.path.basename(self.file_path)}...")
        self.df = pd.read_excel(self.file_path)
        print(f"📊 Model successfully loaded. Shape: {self.df.shape[0]} reactions, {self.df.shape[1]} features.")
        return self.df

    def extract_subsystem_stats(self, top_n: int = 5) -> pd.Series:
        """Returns the top N subsystems ordered by reaction count."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.df["Subsystem"].value_counts().head(top_n)

    def filter_exchange_reactions(self, output_filename: str = "exchange_reactions.xlsx") -> pd.DataFrame:
        """Filters out core exchange reactions and saves them to a clean Excel file."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        exchange_df = self.df[self.df["Subsystem"] == "Exchange reactions"]
        
        # Save output programmatically
        exchange_df.to_excel(output_filename, index=False)
        print(f"💾 Exported {len(exchange_df)} exchange reactions to '{output_filename}'")
        return exchange_df

# ==========================================
# EXECUTION BLOCK (تست ماژول به صورت استاندارد)
# ==========================================
if __name__ == "__main__":
    # آدرس فایل
    TARGET_PATH = r"E:\Biotechnology\all final data\model\modelp2.xlsx"
    
   
    analyzer = MetabolicModelAnalyzer(TARGET_PATH)
    

    analyzer.load_data()
    
    print("\n--- Top 5 Subsystems ---")
    print(analyzer.extract_subsystem_stats())
    
    print("\n--- Filtering Exchange System ---")
    analyzer.filter_exchange_reactions()