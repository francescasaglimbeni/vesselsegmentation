"""
Script per confrontare risultati VECCHI vs NUOVI della pipeline
Verifica se la dual-mask strategy ha migliorato l'accuratezza
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re


class ResultsComparator:
    """Confronta risultati vecchi (senza dual-mask) vs nuovi (con dual-mask)"""
    
    def __init__(self, old_results_dir, new_results_dir):
        self.old_dir = Path(old_results_dir)
        self.new_dir = Path(new_results_dir)
        self.comparison_data = {}
        
    def parse_report_txt(self, report_path):
        """Estrae metriche chiave dal report testuale"""
        metrics = {}
        
        if not os.path.exists(report_path):
            print(f"⚠ Report non trovato: {report_path}")
            return metrics
        
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Estrai metriche con regex
        patterns = {
            'total_branches': r'Total branches:\s*(\d+)',
            'total_length_mm': r'Total tree length:\s*([\d.]+)\s*mm',
            'total_volume_mm3': r'Total tree volume:\s*([\d.]+)\s*mm³',
            'diameter_mean': r'Diameter.*Mean:\s*([\d.]+)\s*mm',
            'max_generation': r'Maximum generation:\s*(\d+)',
            'num_generations': r'Number of generations:\s*(\d+)',
            'central_volume': r'Central:\s*([\d.]+)\s*mm³\s*\((\d+)\s*branches\)',
            'peripheral_volume': r'Peripheral:\s*([\d.]+)\s*mm³\s*\((\d+)\s*branches\)',
            'pc_ratio': r'P/C Ratio:\s*([\d.]+)',
            'tortuosity': r'Tortuosity:\s*([\d.]+)',
            'symmetry': r'Symmetry Index:\s*([\d.]+)',
            'generation_coverage': r'Generation Coverage:\s*([\d.]+)%',
            'fibrosis_score': r'FIBROSIS SCORE:\s*([\d.]+)/100',
            'fibrosis_stage': r'CLASSIFICATION:\s*([^\n]+)',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                if key in ['central_volume', 'peripheral_volume']:
                    metrics[key] = float(match.group(1))
                    metrics[f"{key.split('_')[0]}_branches"] = int(match.group(2))
                else:
                    try:
                        metrics[key] = float(match.group(1))
                    except:
                        metrics[key] = match.group(1).strip()
        
        return metrics
    
    def load_advanced_metrics_json(self, json_path):
        """Carica metriche avanzate da JSON"""
        if not os.path.exists(json_path):
            print(f"⚠ JSON non trovato: {json_path}")
            return {}
        
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def load_branch_metrics_csv(self, csv_path):
        """Carica metriche dei branch da CSV"""
        if not os.path.exists(csv_path):
            print(f"⚠ CSV non trovato: {csv_path}")
            return None
        
        return pd.read_csv(csv_path)
    
    def compare_single_scan(self, scan_name):
        """Confronta risultati per una singola scan"""
        print(f"\n{'='*80}")
        print(f"CONFRONTO RISULTATI: {scan_name}")
        print(f"{'='*80}\n")
        
        # Path vecchi risultati
        old_report = self.old_dir / scan_name / "COMPLETE_ANALYSIS_REPORT.txt"
        old_json = self.old_dir / scan_name / "step4_analysis" / "advanced_metrics.json"
        old_csv = self.old_dir / scan_name / "step4_analysis" / "branch_metrics_complete.csv"
        
        # Path nuovi risultati
        new_report = self.new_dir / scan_name / "COMPLETE_ANALYSIS_REPORT.txt"
        new_json = self.new_dir / scan_name / "step4_analysis" / "advanced_metrics.json"
        new_csv = self.new_dir / scan_name / "step4_analysis" / "branch_metrics_complete.csv"
        
        # Carica dati
        old_metrics = self.parse_report_txt(old_report)
        new_metrics = self.parse_report_txt(new_report)
        
        old_advanced = self.load_advanced_metrics_json(old_json)
        new_advanced = self.load_advanced_metrics_json(new_json)
        
        old_branches = self.load_branch_metrics_csv(old_csv)
        new_branches = self.load_branch_metrics_csv(new_csv)
        
        # Confronto
        comparison = {
            'scan_name': scan_name,
            'old_metrics': old_metrics,
            'new_metrics': new_metrics,
            'old_advanced': old_advanced,
            'new_advanced': new_advanced,
            'old_branches_df': old_branches,
            'new_branches_df': new_branches
        }
        
        self.comparison_data[scan_name] = comparison
        
        # Stampa confronto
        self._print_comparison(comparison)
        
        return comparison
    
    def _print_comparison(self, comp):
        """Stampa confronto formattato"""
        old = comp['old_metrics']
        new = comp['new_metrics']
        
        def format_change(old_val, new_val, is_percentage=False, higher_is_better=True):
            """Formatta la variazione con colore"""
            try:
                old_v = float(old_val)
                new_v = float(new_val)
                diff = new_v - old_v
                perc_change = (diff / old_v * 100) if old_v != 0 else 0
                
                if abs(diff) < 0.001:
                    symbol = "="
                elif (diff > 0 and higher_is_better) or (diff < 0 and not higher_is_better):
                    symbol = "✓"
                else:
                    symbol = "⚠"
                
                if is_percentage:
                    return f"{old_v:.1f}% → {new_v:.1f}% ({diff:+.1f}%) {symbol}"
                else:
                    return f"{old_v:.3f} → {new_v:.3f} ({perc_change:+.1f}%) {symbol}"
            except:
                return f"{old_val} → {new_val}"
        
        print("\n" + "="*80)
        print("METRICHE PRINCIPALI")
        print("="*80)
        
        metrics_to_compare = [
            ("Total branches", 'total_branches', False, True),
            ("Total volume (mm³)", 'total_volume_mm3', False, True),
            ("Max generation", 'max_generation', False, True),
        ]
        
        for label, key, is_perc, higher_better in metrics_to_compare:
            if key in old and key in new:
                print(f"{label:30s}: {format_change(old[key], new[key], is_perc, higher_better)}")
        
        print("\n" + "="*80)
        print("METRICHE CLINICHE CRITICHE")
        print("="*80)
        
        critical_metrics = [
            ("P/C Ratio", 'pc_ratio', False, True),
            ("Peripheral branches", 'peripheral_branches', False, True),
            ("Central branches", 'central_branches', False, True),
            ("Symmetry Index", 'symmetry', False, True),
            ("Tortuosity", 'tortuosity', False, False),
            ("Generation Coverage (%)", 'generation_coverage', True, True),
        ]
        
        for label, key, is_perc, higher_better in critical_metrics:
            if key in old and key in new:
                print(f"{label:30s}: {format_change(old[key], new[key], is_perc, higher_better)}")
        
        print("\n" + "="*80)
        print("FIBROSIS ASSESSMENT")
        print("="*80)
        
        if 'fibrosis_score' in old and 'fibrosis_score' in new:
            print(f"{'Fibrosis Score (0-100)':30s}: {format_change(old['fibrosis_score'], new['fibrosis_score'], False, False)}")
        
        if 'fibrosis_stage' in old and 'fibrosis_stage' in new:
            print(f"{'Classification':30s}: {old['fibrosis_stage']} → {new['fibrosis_stage']}")
        
        # Analisi diametri (se disponibili CSV)
        if comp['old_branches_df'] is not None and comp['new_branches_df'] is not None:
            print("\n" + "="*80)
            print("ANALISI DIAMETRI (dalla dual-mask strategy)")
            print("="*80)
            
            old_df = comp['old_branches_df']
            new_df = comp['new_branches_df']
            
            old_diam = old_df['diameter_mean_mm'].mean()
            new_diam = new_df['diameter_mean_mm'].mean()
            
            print(f"{'Mean diameter (mm)':30s}: {format_change(old_diam, new_diam, False, True)}")
            
            # Analisi per generazione
            if 'generation' in old_df.columns and 'generation' in new_df.columns:
                old_gen_diam = old_df.groupby('generation')['diameter_mean_mm'].mean()
                new_gen_diam = new_df.groupby('generation')['diameter_mean_mm'].mean()
                
                print(f"\nDiametro medio per generazione (primi 5):")
                for gen in range(min(5, len(old_gen_diam))):
                    if gen in old_gen_diam.index and gen in new_gen_diam.index:
                        print(f"  Gen {gen}: {old_gen_diam[gen]:.2f} → {new_gen_diam[gen]:.2f} mm")
    
    def plot_comparison(self, scan_name, output_path=None):
        """Genera grafici di confronto"""
        if scan_name not in self.comparison_data:
            print(f"⚠ Nessun dato di confronto per {scan_name}")
            return
        
        comp = self.comparison_data[scan_name]
        old_df = comp['old_branches_df']
        new_df = comp['new_branches_df']
        
        if old_df is None or new_df is None:
            print("⚠ CSV non disponibili per grafici")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Confronto Risultati: {scan_name}\nVECCHIO (rosso) vs NUOVO (blu)', 
                     fontsize=16, fontweight='bold')
        
        # 1. Distribuzione diametri
        ax = axes[0, 0]
        ax.hist(old_df['diameter_mean_mm'], bins=50, alpha=0.5, label='Vecchio', color='red', edgecolor='black')
        ax.hist(new_df['diameter_mean_mm'], bins=50, alpha=0.5, label='Nuovo', color='blue', edgecolor='black')
        ax.set_xlabel('Diametro medio (mm)')
        ax.set_ylabel('Frequenza')
        ax.set_title('Distribuzione Diametri')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. Distribuzione lunghezze
        ax = axes[0, 1]
        ax.hist(old_df['length_mm'], bins=50, alpha=0.5, label='Vecchio', color='red', edgecolor='black')
        ax.hist(new_df['length_mm'], bins=50, alpha=0.5, label='Nuovo', color='blue', edgecolor='black')
        ax.set_xlabel('Lunghezza (mm)')
        ax.set_ylabel('Frequenza')
        ax.set_title('Distribuzione Lunghezze')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 3. Diametro per generazione
        if 'generation' in old_df.columns and 'generation' in new_df.columns:
            ax = axes[0, 2]
            old_gen = old_df.groupby('generation')['diameter_mean_mm'].agg(['mean', 'std'])
            new_gen = new_df.groupby('generation')['diameter_mean_mm'].agg(['mean', 'std'])
            
            ax.errorbar(old_gen.index, old_gen['mean'], yerr=old_gen['std'], 
                       label='Vecchio', color='red', marker='o', capsize=5, alpha=0.7)
            ax.errorbar(new_gen.index, new_gen['mean'], yerr=new_gen['std'], 
                       label='Nuovo', color='blue', marker='s', capsize=5, alpha=0.7)
            ax.set_xlabel('Generazione Weibel')
            ax.set_ylabel('Diametro medio (mm)')
            ax.set_title('Diametro per Generazione')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # 4. Numero branches per generazione
        if 'generation' in old_df.columns and 'generation' in new_df.columns:
            ax = axes[1, 0]
            old_counts = old_df['generation'].value_counts().sort_index()
            new_counts = new_df['generation'].value_counts().sort_index()
            
            x = np.arange(max(len(old_counts), len(new_counts)))
            width = 0.35
            
            ax.bar(old_counts.index - width/2, old_counts.values, width, 
                  label='Vecchio', color='red', alpha=0.7)
            ax.bar(new_counts.index + width/2, new_counts.values, width, 
                  label='Nuovo', color='blue', alpha=0.7)
            ax.set_xlabel('Generazione')
            ax.set_ylabel('Numero branches')
            ax.set_title('Branches per Generazione')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # 5. Volume per generazione
        if 'generation' in old_df.columns and 'generation' in new_df.columns and 'volume_mm3' in old_df.columns:
            ax = axes[1, 1]
            old_vol = old_df.groupby('generation')['volume_mm3'].sum()
            new_vol = new_df.groupby('generation')['volume_mm3'].sum()
            
            ax.plot(old_vol.index, old_vol.values, 'o-', label='Vecchio', color='red', linewidth=2, markersize=8)
            ax.plot(new_vol.index, new_vol.values, 's-', label='Nuovo', color='blue', linewidth=2, markersize=8)
            ax.set_xlabel('Generazione')
            ax.set_ylabel('Volume totale (mm³)')
            ax.set_title('Volume per Generazione')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # 6. Metriche chiave (barplot)
        ax = axes[1, 2]
        old_m = comp['old_metrics']
        new_m = comp['new_metrics']
        
        metrics_keys = ['pc_ratio', 'symmetry', 'tortuosity']
        metrics_labels = ['P/C Ratio', 'Symmetry', 'Tortuosity']
        
        old_vals = [float(old_m.get(k, 0)) for k in metrics_keys]
        new_vals = [float(new_m.get(k, 0)) for k in metrics_keys]
        
        x_pos = np.arange(len(metrics_labels))
        width = 0.35
        
        ax.bar(x_pos - width/2, old_vals, width, label='Vecchio', color='red', alpha=0.7)
        ax.bar(x_pos + width/2, new_vals, width, label='Nuovo', color='blue', alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics_labels, rotation=45, ha='right')
        ax.set_ylabel('Valore')
        ax.set_title('Metriche Cliniche Chiave')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Grafico salvato: {output_path}")
        else:
            plt.show()
    
    def generate_comparison_report(self, output_path):
        """Genera report testuale completo del confronto"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("REPORT CONFRONTO: VECCHIA vs NUOVA PIPELINE\n")
            f.write("Dual-Mask Strategy: Skeleton da refined, metriche da original\n")
            f.write("="*80 + "\n\n")
            
            for scan_name, comp in self.comparison_data.items():
                f.write(f"\n{'='*80}\n")
                f.write(f"SCAN: {scan_name}\n")
                f.write(f"{'='*80}\n\n")
                
                old = comp['old_metrics']
                new = comp['new_metrics']
                
                f.write("METRICHE CHIAVE:\n")
                f.write("-"*80 + "\n")
                
                for key in ['total_branches', 'total_volume_mm3', 'pc_ratio', 
                           'symmetry', 'tortuosity', 'fibrosis_score']:
                    if key in old and key in new:
                        old_val = old[key]
                        new_val = new[key]
                        try:
                            diff = float(new_val) - float(old_val)
                            perc = (diff / float(old_val) * 100) if float(old_val) != 0 else 0
                            f.write(f"{key:30s}: {old_val:>10} → {new_val:>10} ({perc:+.1f}%)\n")
                        except:
                            f.write(f"{key:30s}: {old_val:>10} → {new_val:>10}\n")
                
                f.write("\n")
        
        print(f"\n✓ Report salvato: {output_path}")


def main():
    """Main function"""
    
    print("\n" + "="*80)
    print(" "*15 + "CONFRONTO VECCHIO vs NUOVO - SINGOLO PAZIENTE")
    print(" "*10 + "Vecchio (no dual-mask) vs Nuovo (con dual-mask)")
    print("="*80)
    
    # ============================================================
    # CONFIGURAZIONE
    # ============================================================
    
    SCAN_NAME = "ID00020637202178344345685"
    
    # Path VECCHI risultati (senza dual-mask strategy)
    OLD_RESULTS_DIR = r"X:\Francesca Saglimbeni\tesi\results\results_OSIC (correct)"
    
    # Path NUOVI risultati (con dual-mask strategy)
    NEW_RESULTS_DIR = r"X:\Francesca Saglimbeni\tesi\vesselsegmentation\airway_segmentation\output_results_with_fibrosis"
    
    # Output
    OUTPUT_DIR = "comparison_results"
    
    # ============================================================
    # ESECUZIONE
    # ============================================================
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\nPaziente: {SCAN_NAME}")
    print(f"Vecchi risultati: {OLD_RESULTS_DIR}")
    print(f"Nuovi risultati:  {NEW_RESULTS_DIR}")
    
    comparator = ResultsComparator(OLD_RESULTS_DIR, NEW_RESULTS_DIR)
    
    # Confronta singola scan
    comparator.compare_single_scan(SCAN_NAME)
    
    # Genera grafici
    plot_path = os.path.join(OUTPUT_DIR, f"{SCAN_NAME}_comparison.png")
    comparator.plot_comparison(SCAN_NAME, output_path=plot_path)
    
    # Genera report
    report_path = os.path.join(OUTPUT_DIR, "comparison_report.txt")
    comparator.generate_comparison_report(report_path)
    
    print("\n" + "="*80)
    print(" "*25 + "CONFRONTO COMPLETATO!")
    print("="*80)
    print(f"\n📁 Output directory: {OUTPUT_DIR}/")
    print(f"📊 Grafico:         {plot_path}")
    print(f"📄 Report:          {report_path}")
    
    # Verifica miglioramenti chiave
    print("\n" + "="*80)
    print("SUMMARY: DUAL-MASK STRATEGY - HA FUNZIONATO?")
    print("="*80)
    
    if SCAN_NAME in comparator.comparison_data:
        comp = comparator.comparison_data[SCAN_NAME]
        old_m = comp['old_metrics']
        new_m = comp['new_metrics']
        
        improvements = []
        regressions = []
        
        # Check P/C ratio
        if 'pc_ratio' in old_m and 'pc_ratio' in new_m:
            old_val = float(old_m['pc_ratio'])
            new_val = float(new_m['pc_ratio'])
            if new_val > old_val * 1.1:  # Miglioramento >10%
                improvements.append(f"✓ P/C Ratio aumentato: {old_val:.3f} → {new_val:.3f}")
            elif new_val < old_val * 0.9:
                regressions.append(f"✗ P/C Ratio diminuito: {old_val:.3f} → {new_val:.3f}")
        
        # Check Symmetry
        if 'symmetry' in old_m and 'symmetry' in new_m:
            old_val = float(old_m['symmetry'])
            new_val = float(new_m['symmetry'])
            if new_val > 0.5 and old_val < 0.5:
                improvements.append(f"✓ Symmetry Index corretto: {old_val:.3f} → {new_val:.3f} (BUG RISOLTO)")
            elif new_val > old_val * 1.1:
                improvements.append(f"✓ Symmetry Index migliorato: {old_val:.3f} → {new_val:.3f}")
            elif new_val < old_val * 0.9:
                regressions.append(f"✗ Symmetry Index peggiorato: {old_val:.3f} → {new_val:.3f}")
        
        # Check peripheral branches
        if 'peripheral_branches' in old_m and 'peripheral_branches' in new_m:
            old_val = int(old_m['peripheral_branches'])
            new_val = int(new_m['peripheral_branches'])
            if new_val > old_val * 1.2:
                improvements.append(f"✓ Branches periferici aumentati: {old_val} → {new_val}")
            elif new_val < old_val * 0.8:
                regressions.append(f"✗ Branches periferici diminuiti: {old_val} → {new_val}")
        
        if improvements:
            print("\n🎉 MIGLIORAMENTI:")
            for imp in improvements:
                print(f"  {imp}")
        
        if regressions:
            print("\n⚠️ REGRESSIONI:")
            for reg in regressions:
                print(f"  {reg}")
        
        if not improvements and not regressions:
            print("\n→ Nessuna variazione significativa rilevata")


if __name__ == "__main__":
    main()
