import json
from datetime import datetime

def run_delta_update(new_scrape_results):
    # 1. Load the history
    with open('market_ledger.json', 'r') as f:
        ledger = json.load(f)

    # 2. Prevent duplicate updates for the same day
    last_entry = ledger[-1]
    if last_entry['date'] == datetime.now().strftime("%Y-%m-%d"):
        print("Today's data already exists. Skipping append.")
        return

    # 3. Create new cumulative entry
    new_entry = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "total_deals": new_scrape_results.get("deals_logged", 0),
        "capacity_mwh": new_scrape_results.get("capacity_mwh", 0),
        "source": "automated_agent",
        "notes": "Agent-derived market update."
    }
    
    # 4. Append to ledger (This is the "Cumulative" part)
    ledger.append(new_entry)
    
    # 5. Save the updated ledger
    with open('market_ledger.json', 'w') as f:
        json.dump(ledger, f, indent=4)
        
    print("Market Ledger updated and appended.")

if __name__ == "__main__":
    # Test execution
    test_data = {"deals_logged": 19, "capacity_mwh": 3150000}
    run_delta_update(test_data)