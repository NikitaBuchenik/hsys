"""Build a PDF draft, or a measured report from successful tests and benchmark files."""
import argparse
import hashlib
import json
import math
import os
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted, Table, TableStyle, Image
from plot import load_measurements, slopes

ROOT=Path(__file__).resolve().parents[1]
REPO='https://github.com/NikitaBuchenik/hsys/tree/main/second'

def fonts():
    options=[(Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts',('times.ttf','timesbd.ttf','consola.ttf')),
             (Path('/usr/share/fonts/truetype/dejavu'),('DejaVuSerif.ttf','DejaVuSerif-Bold.ttf','DejaVuSansMono.ttf'))]
    for folder,names in options:
        if all((folder/n).exists() for n in names):
            for name,file in zip(('Body','Bold','Code'),names): pdfmetrics.registerFont(TTFont(name,str(folder/file)))
            pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Body',boldItalic='Bold')
            return
    raise RuntimeError('Install Times New Roman or DejaVu fonts to build the report')

def validate_results(folder):
    context,rows=load_measurements(folder/'benchmarks.json')
    tree=ET.parse(folder/'tests.xml').getroot()
    cases=tree.findall('.//testcase')
    if not cases or any(c.find('failure') is not None or c.find('error') is not None or c.find('skipped') is not None for c in cases):
        raise ValueError('Tests missing, failed or skipped; a final report cannot be generated')
    shape_cases=[c for c in cases if 'All343Shapes' in c.get('classname','')]
    if len(shape_cases)!=343:
        raise ValueError(f'Expected all 343 parameterized cases, found {len(shape_cases)}')
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    if summary['source_sha256']!=hashlib.sha256((folder/'benchmarks.json').read_bytes()).hexdigest():
        raise ValueError('Plots/summary are stale; rerun plot.py on this benchmark file')
    for name in ('complexity.png','speedup.png'):
        if not (folder/name).is_file(): raise ValueError(f'Missing {name}; run plot.py')
    return context,rows,len(cases)

def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--draft',action='store_true'); mode.add_argument('--results',type=Path)
    parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    fonts(); measured=validate_results(args.results) if args.results else None
    args.output.parent.mkdir(parents=True,exist_ok=True)
    body=ParagraphStyle('Body',fontName='Body',fontSize=14,leading=21,alignment=TA_JUSTIFY,firstLineIndent=12.5*mm,spaceAfter=8)
    center=ParagraphStyle('Center',parent=body,alignment=TA_CENTER,firstLineIndent=0)
    heading=ParagraphStyle('Heading',parent=body,fontName='Bold',firstLineIndent=0,spaceBefore=8,spaceAfter=14)
    small=ParagraphStyle('Small',parent=body,fontSize=10,leading=14,firstLineIndent=0,alignment=0)
    code_style=ParagraphStyle('Code',fontName='Code',fontSize=8.2,leading=11,spaceAfter=12)
    story=[]
    def p(text,style=body): story.append(Paragraph(text,style))
    def h(text): p(text,heading)
    def page(): story.append(PageBreak())
    def code(path,start=None,end=None):
        text=(ROOT/path).read_text(encoding='utf-8')
        if start: text=text[text.index(start):]
        if end: text=text[:text.index(end)]
        lines=[]
        for line in text.strip().splitlines(): lines.extend(textwrap.wrap(line,88,replace_whitespace=False,drop_whitespace=False) or [''])
        story.append(Preformatted('\n'.join(lines),code_style))
    def table(rows,widths):
        t=Table([[Paragraph(escape(str(c)),small) for c in row] for row in rows],colWidths=widths,repeatRows=1)
        t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.4,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eeeeee')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)])); story.append(t)
    p('Министерство науки и высшего образования РФ<br/>Федеральное государственное автономное<br/>образовательное учреждение высшего образования<br/><b>«СИБИРСКИЙ ФЕДЕРАЛЬНЫЙ УНИВЕРСИТЕТ»</b>',center)
    story.append(Spacer(1,8*mm)); p('Институт космических и информационных технологий',center)
    p('Кафедра «Вычислительная техника»',center)
    story.append(Spacer(1,37*mm)); p('<b>ОТЧЕТ О ПРАКТИЧЕСКОЙ РАБОТЕ №2</b>',center)
    p('Умножение матриц на CUDA',center)
    if args.draft: p('ПРОЕКТ: экспериментальная часть ожидает запуска',small)
    story.append(Spacer(1,30*mm))
    for label,identity,name in [('Преподаватель','','С.А. Тарасов'),('Студент','КИ23-06Б 032320481','А.А. Платонов'),('Студент','КИ23-06Б 032321113','Н.С. Бученик')]:
        t=Table([[Paragraph(label,small),Paragraph(identity,small),Paragraph('____________<br/>подпись, дата',small),Paragraph(name,small)]],colWidths=[28*mm,55*mm,33*mm,44*mm]); story.append(t); story.append(Spacer(1,9*mm))
    story.append(Spacer(1,20*mm)); p('Красноярск 2026',center)
    page(); h('ВВЕДЕНИЕ')
    p('Цель работы - закрепить базовые навыки программирования CUDA и освоить работу с двумерными сетками нитей. Требуется реализовать умножение матриц с архитектурой Data + View, проверить результат по Eigen и исследовать время вычислений.')
    p('Разработаны шаблонные классы Data, MatrixView и Matrix. Для умножения используется наивный CUDA-кернел без разделяемой памяти. Каждая нить вычисляет один элемент результата. Исходные данные располагаются в памяти построчно (row-major).')
    p('Проверка включает все 343 сочетания m, n, k из множества {1, 2, 3, 127, 128, 129, 512}. Производительность исследуется для квадратных матриц порядков 16, 32, 64, 128, 256, 512, 1024.')
    p('Полная реализация, тесты, скрипты запуска и построения отчета:')
    p(f'<link href="{REPO}" color="blue">{REPO}</link>',small)
    if args.draft: p('<b>Статус.</b> Этот документ описывает подготовленную реализацию и методику. CUDA-сборка, тесты на GPU и измерения еще не выполнены. Числовые результаты и экспериментальные графики намеренно отсутствуют.')
    page(); h('1 Разработка программного решения'); h('1.1 Владение памятью: Data')
    p('Data непосредственно владеет массивом в глобальной памяти GPU. Конструктор выделяет память через cudaMalloc, деструктор освобождает ее через cudaFree. Копирование Data выполняет глубокую копию, перемещение передает указатель. Оператор копирующего присваивания использует временный объект и swap.')
    code('core/include/data.cuh','    explicit Data','    Data(Data&&')
    p('При ошибке копирования выделенная память освобождается до исключения. Проверяется переполнение размера в байтах. Деструктор не генерирует исключения; ошибки рабочих вызовов CUDA обрабатываются CUDA_CHECK.')
    page(); h('1.2 Представление MatrixView')
    p('MatrixView хранит невладеющий указатель и два размера. Индекс элемента (i, j) равен i · ncols + j. Класс не выделяет и не освобождает память; его объекты передаются в кернел по значению. Тривиальная копируемость проверяется static_assert.')
    code('core/include/matrix_view.cuh','template<class AtomT>')
    p('Входные аргументы кернела имеют тип MatrixView&lt;const AtomT&gt;, поэтому запись в исходные матрицы через этот интерфейс запрещена компилятором. Методы индексации не проверяют границы; кернел проверяет номер строки и столбца до обращения к памяти.')
    page(); h('1.3 Фасад Matrix и оператор умножения')
    p('Matrix объединяет shared_ptr&lt;Data&lt;AtomT&gt;&gt; и собственный MatrixView. Копии Matrix разделяют Data и GPU-память, но имеют разные объекты представления. Перемещение обнуляет представление источника. Обращение к data() перемещенного объекта вызывает исключение.')
    code('core/include/matrix.cuh','    Matrix()','    std::size_t size()')
    code('core/src/matrix.cu','template<class AtomT>\nMatrix<AtomT> operator*','template Matrix<float>')
    p('Оператор проверяет согласованность размеров, создает результат и запускает вычисление. cudaDeviceSynchronize ожидает завершения и выявляет ошибки выполнения. Реализованы инстанцирования для float и double.')
    page(); h('1.4 Двумерный CUDA-кернел')
    code('core/src/matrix.cu','template<class AtomT>\n__global__','template<class AtomT>\nvoid launch_matmul')
    p('Блок содержит 16 × 16 нитей. Размер сетки равен (ceil(n / 16), ceil(m / 16)). Нити за границами результата завершаются без записи, поэтому поддерживаются размеры, не кратные 16, включая 127 и 129.')
    p('Каждая действующая нить последовательно выполняет k умножений и сложений. Всего производится порядка 2mnk арифметических операций; для квадратных матриц - порядка 2n³. Арифметическая сложность составляет O(mnk), объем памяти для входов и выхода - O(mk + kn + mn).')
    p('Наивный алгоритм не использует shared memory и не оптимизирует повторное чтение элементов плитками. Ожидаемое время зависит также от пропускной способности памяти, числа одновременно работающих нитей и накладных расходов запуска.')
    page(); h('2 Модульное тестирование')
    p('Google Test формирует декартово произведение трех наборов размеров через Combine. Эталон рассчитывается как произведение Eigen::MatrixXf. Перед копированием на GPU данные явно преобразуются в row-major, поскольку MatrixXf по умолчанию использует column-major.')
    code('tests/src/matrix_tests.cpp','    const Eigen::MatrixXf a_ref','TEST(Matrix, NonDyadicInputs)')
    p('isApprox использует относительное сравнение норм, а не абсолютную погрешность. Поэтому дополнительно проверяется max|C - Cref| ≤ 10⁻⁵. Основной набор использует значения q/16, q от -8 до 8: произведения и суммы для k ≤ 512 точно представимы в float. Отдельный тест проверяет не двоично-рациональные случайные входы малой амплитуды.')
    p('Дополнительные проверки: единичная и нулевая матрицы, несовместимые и пустые размеры, глубокое копирование Data, разделяемое копирование Matrix, перемещение, переполнение размеров, row-major индексация и double.')
    if measured: p(f'В данном запуске успешно завершены {measured[2]} тестов, включая 343 параметризованных случая. Протокол: tests.xml.')
    else: p('<b>Фактический результат тестов пока отсутствует.</b> Полнота подготовленного набора не означает, что CUDA-реализация уже проверена запуском.')
    page(); h('3 Методика измерения производительности')
    p('Используется Google Benchmark. Для каждого размера выполняются три прогревочных умножения и семь повторов измерений. Для графиков выбирается медиана времени; для времени дополнительно показывается диапазон минимум-максимум. Сборка выполняется в Release, Eigen работает в одном CPU-потоке.')
    p('CUDA-бенчмарк действительно вызывает operator*. Его выделение памяти результата происходит до стартового CUDA Event, а освобождение - после конечного события. События создаются один раз до цикла. Входы копируются на GPU до измерений. Google Benchmark получает длительность между событиями через SetIterationTime.')
    code('benchmarks/src/matrix_benchmark.cpp','        KernelTimer timer;','    } catch')
    p('Для Eigen результат выделяется заранее, затем выполняется c.noalias() = a * b. Время измеряется steady_clock. EIGEN_RUNTIME_NO_MALLOC запрещает динамические выделения внутри измеряемой операции, включая временные буферы. Буферы упаковки размещаются на стеке; лимит стека Windows увеличен до 64 МиБ. Проверка запрета работает и в Release.')
    p('Полученное ускорение относится только к вычислениям. Оно не описывает полное время приложения с передачей данных между CPU и GPU. Малые размеры чувствительны к стоимости запуска и таймера; медиана не устраняет все систематические погрешности.')
    page(); h('4 Результаты вычислительного эксперимента')
    if not measured:
        p('Эксперимент на GPU не выполнен. Планируемая видеокарта по информации исполнителя - RTX 2060. Ее фактические параметры, версия драйвера, CUDA, процессор и времена будут записаны при запуске; сейчас они не представлены как измеренные.')
        p('После успешного запуска scripts/run.ps1 сохраняет tests.xml, benchmarks.json и сведения об окружении. scripts/plot.py читает только реальные записи Google Benchmark, отклоняет ошибки, пропущенные размеры и неполные серии. На их основе создаются таблица, графики времени и ускорения.')
        p('Отчет с экспериментальной частью создается командой scripts/make_report.py --results &lt;папка запуска&gt; --output &lt;report.pdf&gt;. Генерация окончательного отчета запрещена при неуспешных тестах или отсутствии измерений.')
        h('4.1 План анализа')
        p('Для каждого соседнего размера вычисляется показатель p = ln(t₂/t₁) / ln(n₂/n₁). Для кубического роста при удвоении n время увеличивается примерно в 8 раз, p ≈ 3. На малых размерах накладные расходы и неполная загрузка GPU могут дать другое поведение.')
        p('Ускорение определяется как S(n) = tEigen(n) / tCUDA(n). При S &gt; 1 вычислительная часть CUDA быстрее, при S &lt; 1 быстрее Eigen. До получения данных нельзя утверждать наличие ускорения или его максимальное значение.')
    else:
        context,rows,count=measured
        labels={'date':'Дата запуска','gpu':'Видеокарта','compute_capability':'Compute capability',
                'cuda_driver':'CUDA Driver API (код версии)','cuda_runtime':'CUDA Runtime API (код версии)',
                'num_cpus':'Логические процессоры','mhz_per_cpu':'Частота CPU по Google Benchmark, МГц',
                'eigen_threads':'Потоки Eigen'}
        for key,label in labels.items():
            if key in context: p(escape(f'{label}: {context[key]}'),small)
        notes=args.results/'report-notes.txt'
        if notes.is_file():
            for note in notes.read_text(encoding='utf-8-sig').splitlines():
                if note.strip(): p(escape(note),small)
        commit=(args.results/'commit.txt').read_text(encoding='utf-8-sig').strip()
        p('Измеренная версия исходников: '+escape(commit)+'. Замеры относятся именно к указанному коммиту.',small)
        table([['n','CUDA, мс','Eigen, мс','Eigen / CUDA']]+[[r['n'],f"{r['cuda_seconds']*1000:.6f}",f"{r['eigen_seconds']*1000:.6f}",f"{r['speedup']:.3f}"] for r in rows],[25*mm,45*mm,45*mm,45*mm])
        p('В таблице приведены медианы повторных измерений. Исходные данные сохранены в benchmarks.json, протокол проверки - в tests.xml. Сведения о процессоре и частоте отражают данные Google Benchmark, а не измерение частоты во время каждого умножения.',small)
        page(); h('4.1 Графики времени и ускорения')
        for name,caption in [('complexity','Рисунок 1 - Измеренное время вычислений, медиана и минимум-максимум'),('speedup','Рисунок 2 - Ускорение вычислительной части Eigen / CUDA')]:
            story.append(Image(str(args.results/(name+'.png')),width=160*mm,height=94.5*mm)); p(caption,small)
        page(); h('4.2 Анализ результатов')
        cs,es=slopes(rows,'cuda_seconds'),slopes(rows,'eigen_seconds')
        table([['Интервал n','p CUDA','p Eigen']]+[[f"{a['n']} → {b['n']}",f'{c:.3f}',f'{e:.3f}'] for a,b,c,e in zip(rows,rows[1:],cs,es)],[60*mm,50*mm,50*mm])
        p('Локальный показатель p рассчитывается по логарифмическому отношению соседних времен и размеров. Теоретическая арифметическая сложность O(n³) соответствует p = 3; локальные отличия сами по себе не изменяют сложность алгоритма.')
        best=max(rows,key=lambda r:r['speedup']); last=rows[-1]
        p(f"Максимальное измеренное отношение Eigen / CUDA равно {best['speedup']:.3f} при n = {best['n']}. При n = 1024 оно составляет {last['speedup']:.3f}. Значение больше единицы означает выигрыш CUDA только по вычислительной части.")
        faster=[str(r['n']) for r in rows if r['speedup']>1]
        p('CUDA быстрее Eigen на размерах: '+(', '.join(faster) if faster else 'ни на одном из исследованных размеров')+'.')
        p(f'На интервале 512-1024 показатель роста составляет {cs[-1]:.3f} для CUDA и {es[-1]:.3f} для Eigen. На этом интервале оба результата близки к кубической оценке. Для CUDA при переходе от 16 к 64 время уменьшается, поэтому на малых размерах нельзя выводить асимптотику по одному локальному показателю.')
        p('Разброс CUDA на малых размерах заметно выше. Границы на графике - min/max средних времен семи серий, а не доверительный интервал. Сравнение относится к однопоточному Eigen.')
        p('Отклонения могут быть связаны с расходами запуска, загрузкой GPU и кешами CPU. Это гипотезы: для установления причин необходимо аппаратное профилирование.')
    page(); h('ЗАКЛЮЧЕНИЕ')
    p('Подготовлена реализация умножения матриц с разделяемым владением памятью и невладеющим представлением. Двумерная сетка CUDA распределяет элементы результата между нитями. Разделяемая память в кернеле не используется.')
    if measured: p(f'Выполненный запуск подтвердил прохождение {measured[2]} тестов для подготовленного набора данных. Измерены времена CUDA и Eigen, построены графики и рассчитаны ускорение и локальные показатели роста. Выводы ограничены указанным оборудованием, диапазоном размеров и вычислительной частью операции.')
    else: p('Для завершения работы необходимо собрать проект и выполнить тесты и измерения на компьютере с NVIDIA GPU. До этого корректность CUDA-сборки и производительность не заявляются подтвержденными. Настоящий проект отчета не является завершенным экспериментальным отчетом.')
    h('СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ')
    for text in ['1. Тарасов С. А. Практическая работа №2. Материалы задания (work2.pdf).',
                 '2. NVIDIA. CUDA C++ Programming Guide. https://docs.nvidia.com/cuda/cuda-c-programming-guide/',
                 '3. Eigen. DenseBase::isApprox. https://libeigen.gitlab.io/eigen/docs-3.4/classEigen_1_1DenseBase.html',
                 '4. Google Test. Advanced Guide. https://google.github.io/googletest/advanced.html',
                 '5. Google Benchmark. User Guide. https://google.github.io/benchmark/user_guide.html',
                 '6. Eigen. FAQ: runtime allocation checks. https://libeigen.gitlab.io/pages/faq/']:
        p(escape(text),small)
    def footer(canvas,doc):
        if doc.page>1:
            canvas.setFont('Body',12); canvas.drawCentredString(105*mm,15*mm,str(doc.page))
    SimpleDocTemplate(str(args.output),pagesize=(210*mm,297*mm),leftMargin=30*mm,rightMargin=20*mm,
                      topMargin=20*mm,bottomMargin=20*mm,title='Практическая работа №2',author='А.А. Платонов; Н.С. Бученик').build(story,onFirstPage=footer,onLaterPages=footer)
    print(args.output)

if __name__=='__main__': main()
