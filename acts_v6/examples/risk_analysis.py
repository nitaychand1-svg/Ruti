#!/usr/bin/env python3
"""
ACTS v6.0 — Risk Analysis Example

Demonstrates:
1. Existential risk simulation
2. Portfolio stress testing
3. Hedging recommendations
4. VaR analysis
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from acts_v6_complete import ExistentialRiskSimulator, logger


def main():
    logger.info("=" * 70)
    logger.info("ACTS v6.0 — Risk Analysis Example")
    logger.info("=" * 70)
    
    # Initialize risk simulator
    risk_sim = ExistentialRiskSimulator()
    
    # Define portfolio
    portfolio = {
        'BTC': 100000,
        'SPY': 150000,
        'GLD': 50000,
        'USDT': 25000
    }
    
    total_value = sum(portfolio.values())
    logger.info(f"\nPortfolio composition (${total_value:,.0f} total):")
    for asset, value in portfolio.items():
        logger.info(f"  {asset}: ${value:,.0f} ({value/total_value:.1%})")
    
    # ========================================================================
    # SCENARIO ANALYSIS
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("EXISTENTIAL RISK SCENARIOS")
    logger.info("=" * 70)
    
    scenarios_to_test = [
        'solar_flare',
        'cyber_attack',
        'quantum_break',
        'agi_disruption',
        'hyperinflation'
    ]
    
    results = {}
    
    for scenario_name in scenarios_to_test:
        logger.info(f"\n[Scenario: {scenario_name.replace('_', ' ').title()}]")
        logger.info("-" * 70)
        
        result = risk_sim.simulate_scenario(
            scenario_name=scenario_name,
            portfolio=portfolio,
            n_samples=10000
        )
        
        results[scenario_name] = result
        
        # Display results
        logger.info(f"Description: {risk_sim.scenarios[scenario_name]['description']}")
        logger.info(f"Probability: {risk_sim.scenarios[scenario_name]['probability']:.3%}")
        logger.info(f"Expected Loss: ${result['expected_loss']:,.0f}")
        logger.info(f"Std Error: ${result['std_error']:,.0f}")
        logger.info(f"VaR (95%): ${result['var_95']:,.0f}")
        logger.info(f"Survival Probability: {result['survival_probability']:.2%}")
        
        if result['recommended_hedges']:
            logger.info("\nRecommended Hedges:")
            for hedge in result['recommended_hedges']:
                logger.info(f"  - {hedge['asset']}: ${hedge['size']:,.0f}")
                logger.info(f"    Rationale: {hedge['rationale']}")
    
    # ========================================================================
    # COMPARATIVE ANALYSIS
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("COMPARATIVE RISK ANALYSIS")
    logger.info("=" * 70)
    
    logger.info("\nRanked by Expected Loss:")
    logger.info("-" * 70)
    
    sorted_scenarios = sorted(
        results.items(),
        key=lambda x: x[1]['expected_loss']
    )
    
    for i, (name, result) in enumerate(sorted_scenarios, 1):
        logger.info(
            f"{i}. {name.replace('_', ' ').title():20s} "
            f"Loss: ${result['expected_loss']:>12,.0f} "
            f"VaR: ${result['var_95']:>12,.0f}"
        )
    
    # ========================================================================
    # PORTFOLIO RECOMMENDATIONS
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("PORTFOLIO RECOMMENDATIONS")
    logger.info("=" * 70)
    
    # Find most vulnerable scenario
    worst_scenario_name, worst_result = max(
        results.items(),
        key=lambda x: abs(x[1]['expected_loss'])
    )
    
    logger.info(f"\nMost Vulnerable To: {worst_scenario_name.replace('_', ' ').title()}")
    logger.info(f"Potential Loss: ${abs(worst_result['expected_loss']):,.0f} "
                f"({abs(worst_result['expected_loss'])/total_value:.1%} of portfolio)")
    
    # Aggregate hedging recommendations
    all_hedges = {}
    for scenario_name, result in results.items():
        for hedge in result['recommended_hedges']:
            asset = hedge['asset']
            if asset not in all_hedges:
                all_hedges[asset] = {
                    'total_size': 0,
                    'count': 0,
                    'scenarios': []
                }
            all_hedges[asset]['total_size'] += hedge['size']
            all_hedges[asset]['count'] += 1
            all_hedges[asset]['scenarios'].append(scenario_name)
    
    if all_hedges:
        logger.info("\nRecommended Hedging Portfolio:")
        logger.info("-" * 70)
        for asset, info in sorted(all_hedges.items(), key=lambda x: -x[1]['total_size']):
            avg_size = info['total_size'] / info['count']
            logger.info(f"{asset}:")
            logger.info(f"  Average Size: ${avg_size:,.0f} ({avg_size/total_value:.1%} of portfolio)")
            logger.info(f"  Protects Against: {', '.join(s.replace('_', ' ').title() for s in info['scenarios'])}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("RISK ANALYSIS SUMMARY")
    logger.info("=" * 70)
    
    avg_loss = np.mean([r['expected_loss'] for r in results.values()])
    max_loss = min([r['var_95'] for r in results.values()])  # Most extreme VaR
    avg_survival = np.mean([r['survival_probability'] for r in results.values()])
    
    logger.info(f"\nPortfolio Risk Metrics:")
    logger.info(f"  Average Expected Loss (across scenarios): ${avg_loss:,.0f}")
    logger.info(f"  Worst-Case VaR (95%): ${max_loss:,.0f}")
    logger.info(f"  Average Survival Probability: {avg_survival:.2%}")
    logger.info(f"  Risk-Adjusted Portfolio Value: ${total_value + avg_loss:,.0f}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Risk Analysis Complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
