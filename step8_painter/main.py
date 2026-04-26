"""
Main Entry Point - 一键生成所有论文图表
"""
import argparse
import logging
from pathlib import Path
import sys

# 导入各章节的绘图器
from section_b_overall import SectionBPlotter
from section_c_detection_speed import SectionCDetectionSpeed
from section_d_false_alarms import SectionDFalseAlarms
from section_e_tradeoff import SectionETradeoff
from section_f_per_attack import SectionFPerAttack
from section_i_efficiency import SectionIEfficiency

from data_loader import PaperDataLoader, DataLoader
from config import OUTPUT_DIR, FIGURES_DIR, TABLES_DIR

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )


def generate_all_sections(data_loader: PaperDataLoader):
    """生成所有章节的图表"""
    logger.info("="*80)
    logger.info("GENERATING ALL PAPER FIGURES")
    logger.info("="*80)
    
    # 创建新的DataLoader实例（用于新章节）
    new_data_loader = DataLoader()
    
    # Section B: Overall Performance
    logger.info("\n🎨 Section B: Overall Performance")
    section_b = SectionBPlotter(data_loader)
    section_b.generate_all()
    
    # Section C: Detection Speed Analysis
    logger.info("\n🎨 Section C: Detection Speed Analysis")
    section_c = SectionCDetectionSpeed(new_data_loader)
    section_c.generate_all()
    
    # Section D: False Alarm Frequency Analysis
    logger.info("\n🎨 Section D: False Alarm Frequency Analysis")
    section_d = SectionDFalseAlarms(new_data_loader)
    section_d.generate_all()
    
    # Section E: Trade-off Analysis
    logger.info("\n🎨 Section E: Trade-off Analysis")
    section_e = SectionETradeoff()
    section_e.generate_all()
    
    # Section F: Per-Attack-Type Performance
    logger.info("\n🎨 Section F: Per-Attack-Type Performance")
    section_f = SectionFPerAttack()
    section_f.generate_all()
    
    # Section I: Computational Efficiency Analysis
    logger.info("\n🎨 Section I: Computational Efficiency Analysis")
    section_i = SectionIEfficiency()
    section_i.generate_all()
    
    logger.info("\n" + "="*80)
    logger.info("✅ ALL SECTIONS COMPLETE!")
    logger.info("="*80)
    
    # 生成汇总报告
    generate_summary_report()


def generate_section(section: str, data_loader: PaperDataLoader):
    """生成指定章节的图表"""
    section = section.lower()
    new_data_loader = DataLoader()
    
    if section == 'b':
        plotter = SectionBPlotter(data_loader)
        plotter.generate_all()
    elif section == 'c':
        plotter = SectionCDetectionSpeed(new_data_loader)
        plotter.generate_all()
    elif section == 'd':
        plotter = SectionDFalseAlarms(new_data_loader)
        plotter.generate_all()
    elif section == 'e':
        plotter = SectionETradeoff()
        plotter.generate_all()
    elif section == 'f':
        plotter = SectionFPerAttack()
        plotter.generate_all()
    elif section == 'i':
        plotter = SectionIEfficiency()
        plotter.generate_all()
    #     plotter.generate_all()
    # elif section == 'i':
    #     plotter = SectionIPlotter(data_loader)
    #     plotter.generate_all()
    else:
        logger.error(f"Unknown section: {section}")
        logger.info("Available sections: B, C, D, E, F, I")
        sys.exit(1)


def generate_summary_report():
    """生成汇总报告"""
    logger.info("\n📝 Generating summary report...")
    
    summary_dir = OUTPUT_DIR / 'summary'
    summary_dir.mkdir(exist_ok=True)
    
    # 统计生成的文件
    figures_count = len(list(FIGURES_DIR.rglob('*.png')))
    tables_count = len(list(TABLES_DIR.glob('*.csv')))
    
    report_lines = [
        "# Paper Figures Generation Summary",
        "",
        f"**Generated on**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Statistics",
        f"- Total Figures: {figures_count}",
        f"- Total Tables: {tables_count}",
        "",
        "## Output Structure",
        "```",
        "output/",
        "├── figures/",
        "│   ├── section_b/  (Overall Performance)",
        "│   ├── section_c/  (Detection Speed)",
        "│   ├── section_d/  (False Alarms)",
        "│   ├── section_e/  (Trade-off)",
        "│   ├── section_f/  (Per-Attack)",
        "│   └── section_i/  (Efficiency)",
        "└── tables/",
        "    ├── *.csv",
        "    └── *.tex",
        "```",
        "",
        "## Figure Index",
        ""
    ]
    
    # 列出所有图表
    for section_dir in sorted(FIGURES_DIR.iterdir()):
        if section_dir.is_dir():
            section_name = section_dir.name
            report_lines.append(f"### {section_name.upper()}")
            report_lines.append("")
            
            png_files = sorted(section_dir.glob('*.png'))
            for png_file in png_files:
                report_lines.append(f"- `{png_file.name}`")
            report_lines.append("")
    
    # 保存报告
    report_path = summary_dir / 'generation_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"✅ Summary report saved: {report_path}")


def validate_data():
    """验证数据完整性"""
    logger.info("\n🔍 Validating data...")
    
    loader = PaperDataLoader()
    validation = loader.validate_data()
    
    all_valid = all(validation.values())
    
    for key, status in validation.items():
        status_str = "✅" if status else "❌"
        logger.info(f"  {status_str} {key}")
    
    if not all_valid:
        logger.warning("\n⚠️  Some data sources are missing!")
        logger.warning("Please ensure all previous steps (step5, step6, step7, paper_analysis) are complete.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    else:
        logger.info("\n✅ All data sources validated!")
    
    return loader


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Generate all paper figures and tables',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all                 # Generate all sections
  python main.py --section B           # Generate only Section B
  python main.py --section B C D       # Generate multiple sections
  python main.py --validate            # Only validate data
  python main.py --verbose             # Verbose output
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Generate all sections')
    parser.add_argument('--section', nargs='+', choices=['B', 'C', 'D', 'E', 'F', 'I'],
                       help='Generate specific section(s)')
    parser.add_argument('--validate', action='store_true',
                       help='Only validate data sources')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.verbose)
    
    # 验证数据
    data_loader = validate_data()
    
    if args.validate:
        logger.info("\n✅ Validation complete!")
        return
    
    # 生成图表
    if args.all:
        generate_all_sections(data_loader)
    elif args.section:
        for section in args.section:
            generate_section(section, data_loader)
    else:
        # 默认生成所有
        logger.info("No specific section specified, generating all...")
        generate_all_sections(data_loader)
    
    logger.info("\n🎉 Done!")
    logger.info(f"📂 Output directory: {OUTPUT_DIR}")


if __name__ == '__main__':
    import pandas as pd  # 用于时间戳
    main()
