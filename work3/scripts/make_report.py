from pathlib import Path
import argparse,csv,json
from docx import Document
from docx.shared import Cm,Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 summary=json.loads((a.results/'summary.json').read_text(encoding='utf-8'))
 rows=summary['rows'];ctx=summary.get('context',{})
 d=Document();sec=d.sections[0];sec.top_margin=Cm(2);sec.bottom_margin=Cm(2);sec.left_margin=Cm(3);sec.right_margin=Cm(1.5)
 styles=d.styles;styles['Normal'].font.name='Times New Roman';styles['Normal'].font.size=Pt(14)
 for s in ('Title','Heading 1','Heading 2'):styles[s].font.name='Times New Roman'
 p0=d.add_paragraph();p0.alignment=WD_ALIGN_PARAGRAPH.CENTER;p0.add_run('ОТЧЁТ ПО ПРАКТИЧЕСКОЙ РАБОТЕ № 3').bold=True
 p0=d.add_paragraph();p0.alignment=WD_ALIGN_PARAGRAPH.CENTER;p0.add_run('Умножение матриц с использованием разделяемой памяти CUDA')
 d.add_paragraph('Репозиторий: https://github.com/NikitaBuchenik/hsys')
 d.add_heading('1 Цель работы',level=1)
 d.add_paragraph('Реализовать умножение матриц на CUDA с использованием разделяемой памяти, проверить корректность относительно Eigen, измерить время исполнения и сравнить экспериментальные результаты с наивным CUDA-ядром.')
 d.add_heading('2 Реализация',level=1)
 d.add_paragraph('Ядро kernel_matmul_shmem загружает фрагменты матриц размером 16×16 в shared memory, синхронизирует потоки с помощью __syncthreads() и накапливает скалярное произведение. Для частично заполненных плиток выполняется загрузка нулей. MatrixView передаётся в ядро по значению.')
 d.add_paragraph('Оператор operator* использует стратегию SharedMemory. Наивная реализация сохранена для экспериментального сравнения. Для квадратных матриц теоретическая вычислительная сложность обеих реализаций — Θ(n³); тайлинг оптимизирует повторное использование данных, но не меняет асимптотику.')
 d.add_heading('3 Проверка корректности',level=1)
 d.add_paragraph('Автоматические тесты охватывают все 343 сочетания размеров из множества {1, 2, 3, 127, 128, 129, 512} для m, k и n. Результат сравнивается с Eigen::MatrixXf через isApprox с допуском 10⁻⁵. Отдельно проверяются размеры на границах плиток, нулевое внутреннее измерение и несовместимые размеры.')
 d.add_paragraph('Результаты тестирования: см. tests.xml в каталоге измерений. Этот отчёт создаётся после запуска тестов и benchmark; сами тесты должны быть успешно выполнены на целевой системе с CUDA.')
 d.add_heading('4 Методика эксперимента',level=1)
 d.add_paragraph('Входные данные и буферы матриц создаются до измеряемого интервала. CUDA Events размещены вокруг запуска ядра, поэтому выделение памяти, копирование host/device и освобождение памяти не включены во время CUDA-вычислений. Для каждого размера следует использовать несколько повторений и приводить медиану. Параметры среды фиксируются в JSON-отчёте Google Benchmark.')
 d.add_paragraph('Среда выполнения: GPU — '+str(ctx.get('gpu','не записано'))+'; compute capability — '+str(ctx.get('compute_capability','не записано'))+'; CUDA driver — '+str(ctx.get('cuda_driver','не записано'))+'; CUDA runtime — '+str(ctx.get('cuda_runtime','не записано'))+'.')
 d.add_heading('5 Результаты измерений',level=1)
 t=d.add_table(rows=1,cols=4);t.style='Table Grid'
 for c,v in zip(t.rows[0].cells,['n','Наивное ядро, мс','Shared memory, мс','Ускорение, раз']):c.text=v
 for r in rows:
  cells=t.add_row().cells
  for c,v in zip(cells,[r['n'],f"{r['naive_seconds']*1000:.6f}",f"{r['shared_memory_seconds']*1000:.6f}",f"{r['speedup_naive_over_shmem']:.3f}"]):c.text=str(v).replace('.',',')
 for filename,caption in [('complexity.png','Рисунок 1 — Время вычислений в зависимости от размера матрицы'),('speedup.png','Рисунок 2 — Ускорение shared-memory-реализации')]:
  f=a.results/filename
  if f.exists():
   d.add_picture(str(f),width=Cm(15));p1=d.add_paragraph(caption);p1.alignment=WD_ALIGN_PARAGRAPH.CENTER
 d.add_heading('6 Анализ',level=1)
 d.add_paragraph('При увеличении n число арифметических операций растёт пропорционально n³. Shared memory позволяет повторно использовать загруженные элементы матриц и снижать нагрузку на глобальную память. Для малых матриц накладные расходы запуска ядра могут нивелировать выигрыш. Фактическое ускорение зависит от архитектуры GPU, компилятора и размеров матриц; выводы следует делать по графикам и таблице, а не только по теоретической модели.')
 d.add_heading('7 Вывод',level=1)
 d.add_paragraph('Реализовано и проверяется умножение матриц с использованием shared memory. Экспериментальные данные позволяют оценить влияние тайлинга и сопоставить измеренное время с теоретической сложностью Θ(n³). Перед сдачей необходимо приложить протокол успешных тестов и фактические графики, созданные на целевой GPU.')
 a.output.parent.mkdir(parents=True,exist_ok=True);d.save(a.output);print(a.output)
if __name__=='__main__':main()
