package graph;

import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.style.Styler;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.List;
import java.util.stream.Stream;

public class LinePlot {

    public static void main(String[] args) {
        // Exemplo de dados
        List<? extends Number> xData = Arrays.asList(1,2,3,4);

        List<Double> modelData = Arrays.asList(459102.48,230696.34,154770.21,116795.3);
        List<Double> modelError = Arrays.asList(1521.20,748.49,502.23,382.33);

        List<Double> experimentData = Arrays.asList(463489.53, 230384.88,161477.93,121858.38);
        List<Double> experimentError = Arrays.asList(42135.41, 20944.08,14679.81,11078.03);


        try {
            generateGraph(true, "Configuration", "MRT", "ms", "Model", "Experiment", String.valueOf(System.currentTimeMillis()), "./", "graphs", xData, modelData, modelError, experimentData, experimentError);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void generateGraph(boolean hasICBars, String xPrintedName, String yPrintedName, String yPrintedUnit, String modelSeriesName, String experimentSeriesName, String graphFileName, String xmlPath, String graphDirName, List<? extends Number> xData, List<Double> modelData, List<Double> modelError, List<Double> experimentData, List<Double> experimentError) throws IOException {
        Locale.setDefault(Locale.US); // Define a codifica��o dos caracteres para os EUA
        Styler.LegendPosition position = Styler.LegendPosition.OutsideE;

        String yName = yPrintedName + " (" + yPrintedUnit + ")";

        // Cria o gr�fico
        XYChart chart = new XYChartBuilder().width(800).height(600).title(LinePlot.class.getSimpleName())
                .theme(Styler.ChartTheme.Matlab)
                .title("")
                .xAxisTitle(xPrintedName)
                .yAxisTitle(yName)
                .build();

        // Adiciona as s�ries de dados
        if (hasICBars) {
            chart.addSeries(modelSeriesName, xData, modelData, modelError);
            chart.addSeries(experimentSeriesName, xData, experimentData, experimentError);
        } else {
            chart.addSeries(modelSeriesName, xData, modelData);
            chart.addSeries(experimentSeriesName, xData, experimentData);
        }

        // Personaliza o gr�fico
        DecimalFormatSymbols symbols = new DecimalFormatSymbols();
        symbols.setDecimalSeparator('.');
        chart.getStyler().setPlotGridLinesVisible(false);
        DecimalFormat xFormat = new DecimalFormat("#.##########", symbols);
        chart.getStyler().setXAxisDecimalPattern(xFormat.toLocalizedPattern());
        chart.getStyler().setAxisTickLabelsFont(new Font(Font.SANS_SERIF, Font.BOLD, 23));
        chart.getStyler().setAxisTitleFont(new Font(Font.SANS_SERIF, Font.BOLD, 23));
        chart.getStyler().setLegendFont(new Font(Font.SANS_SERIF, Font.BOLD, 23));
        chart.getStyler().setLegendPosition(position);
        chart.getStyler().setErrorBarsColorSeriesColor(true);
        chart.getStyler().setYAxisDecimalPattern("0.00000");

        boolean xHasBigScale = xData.stream().anyMatch(d -> d.doubleValue() > 1.0);
        boolean yHasBigScale = Stream.concat(modelData.stream(), experimentData.stream()).anyMatch(d -> d > 0.01);

        if (xHasBigScale) {
            xFormat = new DecimalFormat("#.##", symbols);
            chart.getStyler().setXAxisDecimalPattern(xFormat.toLocalizedPattern());
        }

        if (yHasBigScale) {
            DecimalFormat yFormat = new DecimalFormat("#.##", symbols);
            chart.getStyler().setYAxisDecimalPattern(yFormat.toLocalizedPattern());
        }

        // Define o caminho para salvar o gr�fico
        String graphPath = xmlPath + graphDirName;

        // Verifica e cria o diret�rio, se necess�rio
        File directory = new File(graphPath);
        if (!directory.exists()) {
            directory.mkdirs();
        }

        // Salva o gr�fico
        BitmapEncoder.saveBitmapWithDPI(chart, graphPath + "/" + graphFileName, BitmapEncoder.BitmapFormat.PNG, 400);

        System.out.println("O gr�fico foi gerado com sucesso no diret�rio:\n" + graphPath);



        System.out.println("Momento de Fim: " + new SimpleDateFormat("HH:mm:ss").format(new Date()));
    }




}
