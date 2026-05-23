use pairset_combo_planner::{plan_combos, PlannerRequest};
use std::io::{self, Read};

fn main() {
    if let Err(err) = run() {
        eprintln!("pairset-combo-planner: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let request: PlannerRequest = serde_json::from_str(&input)?;
    let response = plan_combos(&request)?;
    serde_json::to_writer(std::io::stdout(), &response)?;
    Ok(())
}
