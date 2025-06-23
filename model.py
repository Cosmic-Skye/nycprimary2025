import math
from datetime import datetime

def calculate_nyc_primary_comprehensive():
    """
    Model for NYC Democratic Primary 2025
    Incorporates all available data with granular demographic and geographic analysis
    """
    
    # ========== CORE DATA INPUTS ==========
    
    # Enhanced Polling Data with more detail
    POLLS = [
        {
            'name': 'Emerson',
            'date': 'June 18-20',
            'days_old': 3,
            'credibility': 0.95,
            'sample_size': 833,
            'margin_error': 3.3,
            'first_choice': {'mamdani': 32.0, 'cuomo': 35.0, 'lander': 13.0, 'adams': 8.0},
            'rcv_final': {'mamdani': 52.0, 'cuomo': 48.0},
            'early_voters': {'mamdani': 41.0, 'cuomo': 31.0}
        },
        {
            'name': 'Marist',
            'date': 'June 11-16',
            'days_old': 7,
            'credibility': 0.98,
            'sample_size': 644,
            'margin_error': 3.9,
            'first_choice': {'mamdani': 27.0, 'cuomo': 38.0, 'lander': 7.0, 'adams': 7.0},
            'rcv_final': {'mamdani': 45.0, 'cuomo': 55.0},
            'early_voters': None
        },
        {
            'name': 'Manhattan Institute',
            'date': 'June 11-16',
            'days_old': 7,
            'credibility': 0.70,
            'sample_size': 606,
            'margin_error': 3.9,
            'first_choice': {'mamdani': 30.0, 'cuomo': 43.0, 'lander': 11.0, 'adams': 8.0},
            'rcv_final': {'mamdani': 44.0, 'cuomo': 56.0},
            'early_voters': None
        }
    ]
    
    # Borough-Specific Early Vote Data (NYC BOE)
    EARLY_VOTE_BY_BOROUGH = {
        'Manhattan': {'total': 122642, 'pct_of_total': 31.8, 'mamdani_est': 0.45, 'cuomo_est': 0.28},
        'Brooklyn': {'total': 142724, 'pct_of_total': 37.1, 'mamdani_est': 0.44, 'cuomo_est': 0.29},
        'Queens': {'total': 75778, 'pct_of_total': 19.7, 'mamdani_est': 0.38, 'cuomo_est': 0.34},
        'Bronx': {'total': 30816, 'pct_of_total': 8.0, 'mamdani_est': 0.31, 'cuomo_est': 0.42},
        'Staten Island': {'total': 12367, 'pct_of_total': 3.2, 'mamdani_est': 0.28, 'cuomo_est': 0.45}
    }
    
    TOTAL_EARLY_VOTES = 385327
    
    # Demographic Breakdown (estimated from reporting)
    AGE_DEMOGRAPHICS = {
        '18-24': {'pct_early': 0.10, 'pct_historical': 0.05, 'mamdani_support': 0.65, 'heat_impact': -0.05},
        '25-34': {'pct_early': 0.25, 'pct_historical': 0.07, 'mamdani_support': 0.60, 'heat_impact': -0.05},
        '35-49': {'pct_early': 0.30, 'pct_historical': 0.25, 'mamdani_support': 0.45, 'heat_impact': -0.15},
        '50-64': {'pct_early': 0.25, 'pct_historical': 0.35, 'mamdani_support': 0.35, 'heat_impact': -0.25},
        '65+': {'pct_early': 0.10, 'pct_historical': 0.28, 'mamdani_support': 0.25, 'heat_impact': -0.45}
    }
    
    # Weather Model (NWS)
    WEATHER_FORECAST = {
        'high_temp_f': 101,
        'heat_index_f': 106,
        'humidity': 65,
        'is_record': True,  # First 100F since 2012
        'poll_sites_no_ac': 600,
        'total_poll_sites': 1213
    }
    
    # Historical Turnout Data
    HISTORICAL_DATA = {
        '2021_primary': 943996,
        '2017_primary': 650361,
        '2013_primary': 691801,
        '2021_early_vote': 191239,
        '2021_eday_vote': 752757
    }
    
    # Cross-Endorsement Impact
    ENDORSEMENTS = {
        'mamdani_lander': {'transfer_efficiency': 0.75, 'lander_first_choice': 0.13},
        'mamdani_blake': {'transfer_efficiency': 0.70, 'blake_first_choice': 0.02},
        'dont_rank_cuomo': {'impact': 0.05}  # WFP campaign
    }
    
    
    # ========== STAGE 1: WEIGHTED POLLING BASELINE ==========
    
    def calculate_poll_weight(poll):
        """Calculate poll weight based on recency, credibility, and sample size"""
        time_decay = 0.95 ** poll['days_old']
        sample_weight = min(1.0, poll['sample_size'] / 1000)
        return poll['credibility'] * time_decay * sample_weight
    
    # Calculate weighted RCV average
    total_weight = 0
    weighted_rcv_sum = 0
    weighted_first_sum = 0
    
    for poll in POLLS:
        weight = calculate_poll_weight(poll)
        weighted_rcv_sum += poll['rcv_final']['mamdani'] * weight
        weighted_first_sum += poll['first_choice']['mamdani'] * weight
        total_weight += weight
    
    baseline_rcv_probability = weighted_rcv_sum / total_weight
    baseline_first_choice = weighted_first_sum / total_weight
    
    # ========== STAGE 2: EARLY VOTE ANALYSIS ==========
    
    # Calculate borough-weighted early vote advantage
    mamdani_early_votes = 0
    cuomo_early_votes = 0
    
    for borough, data in EARLY_VOTE_BY_BOROUGH.items():
        mamdani_early_votes += data['total'] * data['mamdani_est']
        cuomo_early_votes += data['total'] * data['cuomo_est']
    
    mamdani_early_advantage = mamdani_early_votes - cuomo_early_votes
    
    # Age-based early vote composition analysis
    youth_early_vote_share = (AGE_DEMOGRAPHICS['18-24']['pct_early'] + 
                              AGE_DEMOGRAPHICS['25-34']['pct_early'])
    youth_overperformance = youth_early_vote_share / (AGE_DEMOGRAPHICS['18-24']['pct_historical'] + 
                                                      AGE_DEMOGRAPHICS['25-34']['pct_historical'])
    
    # ========== STAGE 3: ELECTION DAY TURNOUT MODEL ==========
    
    # Base turnout projection
    base_turnout_projection = HISTORICAL_DATA['2021_primary'] * 1.08  # 8% increase for engagement
    
    # Heat impact by demographic
    total_heat_suppression = 0
    for age_group, demo in AGE_DEMOGRAPHICS.items():
        group_size = (base_turnout_projection - TOTAL_EARLY_VOTES) * demo['pct_historical']
        heat_loss = group_size * demo['heat_impact']
        total_heat_suppression += heat_loss
    
    # Additional heat factors
    no_ac_impact = (WEATHER_FORECAST['poll_sites_no_ac'] / WEATHER_FORECAST['total_poll_sites']) * 0.10
    extreme_heat_multiplier = 1.2 if WEATHER_FORECAST['high_temp_f'] >= 100 else 1.0
    
    total_heat_suppression *= (1 + no_ac_impact) * extreme_heat_multiplier
    
    projected_eday_turnout = (base_turnout_projection - TOTAL_EARLY_VOTES) + total_heat_suppression
    projected_total_turnout = TOTAL_EARLY_VOTES + projected_eday_turnout
    
    # ========== STAGE 4: DEMOGRAPHIC VOTE MODELING ==========
    
    def calculate_eday_votes_by_demo():
        """Model E-Day vote by age group accounting for heat"""
        eday_votes = {}
        mamdani_eday_total = 0
        cuomo_eday_total = 0
        
        for age_group, demo in AGE_DEMOGRAPHICS.items():
            # Adjust historical percentage for heat impact
            heat_adjusted_pct = demo['pct_historical'] * (1 + demo['heat_impact'])
            # Normalize percentages
            total_heat_adjusted = sum(d['pct_historical'] * (1 + d['heat_impact']) 
                                     for d in AGE_DEMOGRAPHICS.values())
            normalized_pct = heat_adjusted_pct / total_heat_adjusted
            
            group_eday_votes = projected_eday_turnout * normalized_pct
            mamdani_votes = group_eday_votes * demo['mamdani_support']
            cuomo_votes = group_eday_votes * (1 - demo['mamdani_support'])
            
            mamdani_eday_total += mamdani_votes
            cuomo_eday_total += cuomo_votes
            
            eday_votes[age_group] = {
                'total': group_eday_votes,
                'mamdani': mamdani_votes,
                'cuomo': cuomo_votes
            }
        
        return eday_votes, mamdani_eday_total, cuomo_eday_total
    
    eday_breakdown, mamdani_eday, cuomo_eday = calculate_eday_votes_by_demo()
    
    # ========== STAGE 5: RANKED CHOICE VOTING SIMULATION ==========
    
    # Apply momentum factor to vote totals
    momentum_boost = 0.025  # 2.5% momentum for Mamdani based on polling trends
    
    # Apply boost to Mamdani's votes
    mamdani_total_adjusted = (mamdani_early_votes + mamdani_eday) * (1 + momentum_boost)
    cuomo_total_adjusted = (cuomo_early_votes + cuomo_eday) * (1 - momentum_boost * 0.7)  # Cuomo loses some to Mamdani
    
    # First choice projections
    first_choice_totals = {
        'mamdani': mamdani_total_adjusted,
        'cuomo': cuomo_total_adjusted,
        'lander': projected_total_turnout * 0.12,  # From polling
        'adams': projected_total_turnout * 0.08,
        'others': projected_total_turnout * 0.10
    }
    
    # RCV transfers based on endorsements
    def simulate_rcv():
        """Simulate ranked choice voting rounds"""
        votes = first_choice_totals.copy()
        eliminated = []
        rounds = []
        
        # Round 1 - Initial count
        round_data = {
            'round': 1,
            'votes': votes.copy(),
            'eliminated': None,
            'transfers': {}
        }
        rounds.append(round_data)
        
        round_num = 2
        while len([k for k, v in votes.items() if v > 0 and k not in eliminated]) > 2:
            # Find lowest vote-getter
            active_candidates = {k: v for k, v in votes.items() if k not in eliminated and v > 0}
            lowest = min(active_candidates.keys(), key=lambda x: active_candidates[x])
            
            # Track transfers
            transfers = {}
            eliminated_votes = votes[lowest]
            
            # Transfer votes
            if lowest == 'lander':
                # Lander-Mamdani endorsement
                transfer_to_mamdani = eliminated_votes * ENDORSEMENTS['mamdani_lander']['transfer_efficiency']
                transfer_to_cuomo = eliminated_votes * 0.15
                transfer_exhausted = eliminated_votes * 0.10
                votes['mamdani'] += transfer_to_mamdani
                votes['cuomo'] += transfer_to_cuomo
                transfers = {
                    'mamdani': transfer_to_mamdani,
                    'cuomo': transfer_to_cuomo,
                    'exhausted': transfer_exhausted
                }
            elif lowest == 'adams':
                # Adams voters split
                transfer_to_cuomo = eliminated_votes * 0.45
                transfer_to_mamdani = eliminated_votes * 0.35
                transfer_exhausted = eliminated_votes * 0.20
                votes['cuomo'] += transfer_to_cuomo
                votes['mamdani'] += transfer_to_mamdani
                transfers = {
                    'cuomo': transfer_to_cuomo,
                    'mamdani': transfer_to_mamdani,
                    'exhausted': transfer_exhausted
                }
            else:
                # Generic progressive transfer pattern
                transfer_to_mamdani = eliminated_votes * 0.60
                transfer_to_cuomo = eliminated_votes * 0.30
                transfer_exhausted = eliminated_votes * 0.10
                votes['mamdani'] += transfer_to_mamdani
                votes['cuomo'] += transfer_to_cuomo
                transfers = {
                    'mamdani': transfer_to_mamdani,
                    'cuomo': transfer_to_cuomo,
                    'exhausted': transfer_exhausted
                }
            
            votes[lowest] = 0
            eliminated.append(lowest)
            
            round_data = {
                'round': round_num,
                'votes': votes.copy(),
                'eliminated': lowest,
                'transfers': transfers
            }
            rounds.append(round_data)
            round_num += 1
        
        # Final round percentages
        total_final = votes['mamdani'] + votes['cuomo']
        mamdani_final_pct = (votes['mamdani'] / total_final) * 100
        
        return mamdani_final_pct, rounds
    
    rcv_simulation_result, rcv_rounds = simulate_rcv()
    
    # ========== STAGE 6: FINAL CALCULATIONS ==========
    
    # Use RCV result as final vote share
    final_mamdani_vote_share = rcv_simulation_result
    
    # Polling error uncertainty
    historical_polling_error = 3.5  # Average NYC primary polling error
    
    # Convert vote share to win probability using normal distribution
    # Mean = projected vote share, SD = polling error
    margin = final_mamdani_vote_share - 50.0
    z_score = margin / historical_polling_error
    
    # Better approximation of normal CDF using error function
    # erf approximation: erf(x) ≈ sign(x) * sqrt(1 - exp(-x² * (4/π + ax²) / (1 + ax²)))
    # where a ≈ 0.147
    def erf_approx(x):
        a = 0.147
        x_squared = x * x
        numerator = 4.0/math.pi + a * x_squared
        denominator = 1.0 + a * x_squared
        return math.copysign(1, x) * math.sqrt(1.0 - math.exp(-x_squared * numerator / denominator))
    
    # Normal CDF: Φ(z) = 0.5 * (1 + erf(z/sqrt(2)))
    final_mamdani_probability = 50.0 * (1.0 + erf_approx(z_score / math.sqrt(2.0)))
    
    # Calculate confidence interval for VOTE SHARE (not win probability)
    # For 95% CI, use 1.96 standard deviations
    vote_share_lower = final_mamdani_vote_share - 1.96 * historical_polling_error
    vote_share_upper = final_mamdani_vote_share + 1.96 * historical_polling_error
    
    # Ensure bounds are sensible
    vote_share_lower = max(40.0, min(60.0, vote_share_lower))
    vote_share_upper = max(40.0, min(60.0, vote_share_upper))
    
    # Recalculate win probability based on proper CI
    # If lower bound > 50%, win probability should be > 97.5%
    if vote_share_lower >= 50.0:
        # Calculate how many SDs above 50% the lower bound is
        z_score_lower = (vote_share_lower - 50.0) / historical_polling_error
        # Win probability is 1 - P(Z < -z_score_lower) where z_score_lower corresponds to 2.5% tail
        final_mamdani_probability = 100.0 - (2.5 * math.exp(-z_score_lower))  # Approximation for high probability
        final_mamdani_probability = min(99.9, final_mamdani_probability)
    
    # ========== OUTPUT COMPREHENSIVE RESULTS ==========
    
    print("=" * 80)
    print("NYC DEMOCRATIC MAYORAL PRIMARY 2025: PREDICTION MODEL")
    print("=" * 80)
    print(f"\nModel Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📊 POLLING ANALYSIS")
    print(f"  Weighted RCV Baseline: Mamdani {baseline_rcv_probability:.1f}%")
    print(f"  Weighted First Choice: Mamdani {baseline_first_choice:.1f}%")
    
    print("\n🗳️ EARLY VOTING BREAKDOWN")
    print(f"  Total Early Votes: {TOTAL_EARLY_VOTES:,}")
    print(f"  Mamdani Early Votes: {int(mamdani_early_votes):,} ({mamdani_early_votes/TOTAL_EARLY_VOTES*100:.1f}%)")
    print(f"  Cuomo Early Votes: {int(cuomo_early_votes):,} ({cuomo_early_votes/TOTAL_EARLY_VOTES*100:.1f}%)")
    print(f"  Mamdani Net Advantage: +{int(mamdani_early_advantage):,} votes")
    print(f"  Youth Vote Surge: {youth_overperformance:.1f}x historical rate")
    
    print("\n🌡️ HEAT IMPACT ANALYSIS")
    print(f"  Forecast High: {WEATHER_FORECAST['high_temp_f']}°F (Heat Index: {WEATHER_FORECAST['heat_index_f']}°F)")
    print(f"  Sites Without AC: {WEATHER_FORECAST['poll_sites_no_ac']}/{WEATHER_FORECAST['total_poll_sites']} ({WEATHER_FORECAST['poll_sites_no_ac']/WEATHER_FORECAST['total_poll_sites']*100:.1f}%)")
    print(f"  Total Vote Suppression: {int(abs(total_heat_suppression)):,} fewer voters")
    print("  Impact by Age Group:")
    for age, data in AGE_DEMOGRAPHICS.items():
        print(f"    {age}: {data['heat_impact']*100:.0f}% reduction")
    
    print("\n📈 TURNOUT PROJECTIONS")
    print(f"  Historical 2021 Turnout: {HISTORICAL_DATA['2021_primary']:,}")
    print(f"  Base 2025 Projection: {int(base_turnout_projection):,}")
    print(f"  Heat-Adjusted E-Day Turnout: {int(projected_eday_turnout):,}")
    print(f"  Final Total Turnout: {int(projected_total_turnout):,}")
    
    print("\n🗳️ ELECTION DAY VOTE MODEL")
    for age, votes in eday_breakdown.items():
        print(f"  {age}: {int(votes['total']):,} votes")
        print(f"    → Mamdani: {int(votes['mamdani']):,} ({votes['mamdani']/votes['total']*100:.1f}%)")
        print(f"    → Cuomo: {int(votes['cuomo']):,} ({votes['cuomo']/votes['total']*100:.1f}%)")
    
    print(f"\n  E-Day Totals:")
    print(f"    Mamdani: {int(mamdani_eday):,} votes")
    print(f"    Cuomo: {int(cuomo_eday):,} votes")
    print(f"    Cuomo needs {(mamdani_early_advantage/projected_eday_turnout)*100:.1f}% E-Day margin to tie")
    
    print("\n🔄 RANKED CHOICE VOTING SIMULATION")
    
    # Print each round
    for round_data in rcv_rounds:
        print(f"\n  Round {round_data['round']}:")
        
        # Sort candidates by votes for this round
        active_votes = {k: v for k, v in round_data['votes'].items() if v > 0}
        total_active_votes = sum(active_votes.values())
        for candidate, votes in sorted(active_votes.items(), key=lambda x: x[1], reverse=True):
            pct = (votes/total_active_votes)*100
            print(f"    {candidate.title()}: {int(votes):,} ({pct:.1f}%)")
        
        if round_data['eliminated']:
            print(f"\n    ❌ {round_data['eliminated'].title()} eliminated")
            if round_data['transfers']:
                print("    Transfers:")
                for recipient, votes in round_data['transfers'].items():
                    if votes > 0:
                        print(f"      → {recipient.title()}: {int(votes):,} votes")
    
    # Final result
    print(f"\n  FINAL RESULT:")
    final_round = rcv_rounds[-1]
    final_votes = {k: v for k, v in final_round['votes'].items() if v > 0}
    total_remaining = sum(final_votes.values())
    for candidate, votes in sorted(final_votes.items(), key=lambda x: x[1], reverse=True):
        pct = (votes/total_remaining)*100
        print(f"    {candidate.title()}: {int(votes):,} ({pct:.1f}%)")
    
    print("\n🌊 FACTORS INCORPORATED")
    print(f"  Momentum adjustment: +{momentum_boost*100:.1f}% for Mamdani (based on polling trends)")
    print(f"  Cross-Endorsement: {ENDORSEMENTS['mamdani_lander']['transfer_efficiency']*100:.0f}% Lander→Mamdani transfers")
    
    print("\n" + "=" * 80)
    print("🎯 FINAL PREDICTION")
    print("=" * 80)
    
    print(f"\nMAMDANI WIN PROBABILITY: {final_mamdani_probability:.1f}%")
    print(f"CUOMO WIN PROBABILITY: {100-final_mamdani_probability:.1f}%")
    
    print("\nPROJECTED FINAL VOTE SHARE:")
    print(f"  Zohran Mamdani: {final_mamdani_vote_share:.1f}%")
    print(f"  Andrew Cuomo: {100-final_mamdani_vote_share:.1f}%")
    print(f"  95% Confidence Interval: {vote_share_lower:.1f}% - {vote_share_upper:.1f}% (Mamdani vote share)")
    
    # Calculate margin of victory range
    margin_lower = (vote_share_lower - 50) * 2  # Convert to two-candidate margin
    margin_upper = (vote_share_upper - 50) * 2
    margin_votes_lower = int((margin_lower/100) * projected_total_turnout)
    margin_votes_upper = int((margin_upper/100) * projected_total_turnout)
    
    print(f"\nVICTORY MARGIN:")
    print(f"  Projected: {(final_mamdani_vote_share-50)*2:.1f}% (~{int((final_mamdani_vote_share-50)/100 * projected_total_turnout * 2):,} votes)")
    print(f"  95% Confidence Range: {margin_lower:.1f}% - {margin_upper:.1f}%")
    print(f"  In votes: {margin_votes_lower:,} - {margin_votes_upper:,} votes")
    
    return final_mamdani_probability

# Run the comprehensive model
if __name__ == "__main__":
    result = calculate_nyc_primary_comprehensive()