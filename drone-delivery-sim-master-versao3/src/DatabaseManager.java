import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.LinkedBlockingQueue;


public class DatabaseManager {
    private static class MetricValue {
        String proxyName;
        String name;
        Float value;
        MetricValue(String proxyName, String name, Float value) {
            this.proxyName = proxyName;
            this.name = name;
            this.value = value;
        }
    }

    private final Map<String, List<Float>> metrics = new ConcurrentHashMap<>();
    private final BlockingQueue<MetricValue> queue = new LinkedBlockingQueue<>();

    private static DatabaseManager instance;

    private DatabaseManager() {}

    public static DatabaseManager getInstance() {
        if (instance == null) {
            instance = new DatabaseManager();
        }
        return instance;
    }

    // --- NOVO: modo síncrono ---
    public void addMetric(String proxyName, String metricName, Float value, long timestamp) {
        queue.offer(new MetricValue(proxyName, metricName, value));
        processQueue(); // ✅ Processa imediatamente tudo que estiver na fila
    }

    private void processQueue() {
        MetricValue mv;
        while ((mv = queue.poll()) != null) {
            List<Float> values = metrics.computeIfAbsent(mv.proxyName + "_" + mv.name,
                    k -> new ArrayList<>());
            values.add(mv.value);
        }
    }

    public Float getAverageBelowP95MetricValue(String proxyName, String metricName) {
        String key = proxyName + "_" + metricName;
        List<Float> values = metrics.remove(key); // Remove do mapa e retorna a lista

        if (values == null || values.isEmpty()) {
            return 0.0F;
        }

        // Criamos uma cópia ordenada (para não interferir em outras operações)
        List<Float> sortedValues = new ArrayList<>(values);
        Collections.sort(sortedValues);

        int size = sortedValues.size();

        // Índice do percentil 95
        double p95Index = 0.95 * (size - 1);
        int lowerIndex = (int) Math.floor(p95Index);
        int upperIndex = (int) Math.ceil(p95Index);

        float p95;
        if (lowerIndex == upperIndex) {
            p95 = sortedValues.get(lowerIndex);
        } else {
            // Interpolação linear entre os dois valores
            float lowerValue = sortedValues.get(lowerIndex);
            float upperValue = sortedValues.get(upperIndex);
            p95 = lowerValue + (upperValue - lowerValue) * (float) (p95Index - lowerIndex);
        }

        // Pega só os valores abaixo do p95
        List<Float> belowP95 = new ArrayList<>();
        for (Float v : sortedValues) {
            if (v < p95) {
                belowP95.add(v);
            }
        }

        if (belowP95.isEmpty()) {
            return 0.0F; // evita divisão por zero
        }

        float sum = 0;
        for (Float v : belowP95) {
            sum += v;
        }

        return sum / belowP95.size();
    }

    public Float getAverageBelowP90MetricValue(String proxyName, String metricName) {
        String key = proxyName + "_" + metricName;
        List<Float> values = metrics.remove(key); // Remove do mapa e retorna a lista

        if (values == null || values.isEmpty()) {
            return 0.0F;
        }

        // Criamos uma cópia ordenada (para não interferir em outras operações)
        List<Float> sortedValues = new ArrayList<>(values);
        Collections.sort(sortedValues);

        int size = sortedValues.size();

        // Índice do percentil 90
        double p90Index = 0.90 * (size - 1);
        int lowerIndex = (int) Math.floor(p90Index);
        int upperIndex = (int) Math.ceil(p90Index);

        float p90;
        if (lowerIndex == upperIndex) {
            p90 = sortedValues.get(lowerIndex);
        } else {
            // Interpolação linear entre os dois valores
            float lowerValue = sortedValues.get(lowerIndex);
            float upperValue = sortedValues.get(upperIndex);
            p90 = lowerValue + (upperValue - lowerValue) * (float) (p90Index - lowerIndex);
        }

        // Pega só os valores abaixo do p90
        List<Float> belowP90 = new ArrayList<>();
        for (Float v : sortedValues) {
            if (v < p90) {
                belowP90.add(v);
            }
        }

        if (belowP90.isEmpty()) {
            return 0.0F; // evita divisão por zero
        }

        float sum = 0;
        for (Float v : belowP90) {
            sum += v;
        }

        return sum / belowP90.size();
    }

    public Float getAverageBelowP75MetricValue(String proxyName, String metricName) {
        String key = proxyName + "_" + metricName;
        List<Float> values = metrics.remove(key); // Remove do mapa e retorna a lista

        if (values == null || values.isEmpty()) {
            return 0.0F;
        }

        // Criamos uma cópia ordenada (para não interferir em outras operações)
        List<Float> sortedValues = new ArrayList<>(values);
        Collections.sort(sortedValues);

        int size = sortedValues.size();

        // Índice do percentil 75
        double p75Index = 0.75 * (size - 1);
        int lowerIndex = (int) Math.floor(p75Index);
        int upperIndex = (int) Math.ceil(p75Index);

        float p75;
        if (lowerIndex == upperIndex) {
            p75 = sortedValues.get(lowerIndex);
        } else {
            // interpolação linear entre os dois valores
            float lowerValue = sortedValues.get(lowerIndex);
            float upperValue = sortedValues.get(upperIndex);
            p75 = lowerValue + (upperValue - lowerValue) * (float) (p75Index - lowerIndex);
        }

        // Pega só os valores abaixo do p75
        List<Float> belowP75 = new ArrayList<>();
        for (Float v : sortedValues) {
            if (v < p75) {
                belowP75.add(v);
            }
        }

        if (belowP75.isEmpty()) {
            return 0.0F; // evita divisão por zero
        }

        float sum = 0;
        for (Float v : belowP75) {
            sum += v;
        }

        return sum / belowP75.size();
    }

    public Float getMedianMetricValue(String proxyName, String metricName) {
        String key = proxyName + "_" + metricName;
        List<Float> values = metrics.remove(key); // Remove do mapa e retorna a lista

        if (values == null || values.isEmpty()) {
            return 0.0F;
        }

        // Criamos uma cópia para ordenar, para não interferir nas operações de escrita
        List<Float> sortedValues = new ArrayList<>(values);
        Collections.sort(sortedValues);

        int size = sortedValues.size();

        // cálculo da mediana
        if (size % 2 == 1) {
            // tamanho ímpar → elemento do meio
            return sortedValues.get(size / 2);
        } else {
            // tamanho par → média dos dois elementos centrais
            float middle1 = sortedValues.get((size / 2) - 1);
            float middle2 = sortedValues.get(size / 2);
            return (middle1 + middle2) / 2.0f;
        }
    }

    public Float getAverageMetricValue(String proxyName, String metricName) {
        String key = proxyName + "_" + metricName;
        List<Float> values = metrics.remove(key); // Remove do mapa e retorna a lista

        if (values == null || values.isEmpty()) {
            return 0.0F;
        }


        // Criamos uma cópia para ordenar, para não interferir nas operações de escrita
        List<Float> sortedValues = new CopyOnWriteArrayList<>(values);

        int size = sortedValues.size();
        float sum = 0;
        for (Float value : sortedValues) sum += value;

        return sum / size;
    }

    public Float getAverageMetricValue30Samples(String proxyName, String metricName) {
        String key = proxyName + "_" + metricName;
        List<Float> values = metrics.get(key); // pega a lista sem remover

        if (values == null || values.isEmpty()) {
            return 0.0F;
        }

        // Mantém apenas as últimas 30 amostras
        int maxSamples = 30;
        if (values.size() > maxSamples) {
            int excess = values.size() - maxSamples;
            values.subList(0, excess).clear(); // remove as mais antigas
        }

        // Calcula a média
        float sum = 0;
        for (Float value : values) {
            sum += value;
        }

        return sum / values.size();
    }

    public long getSizeMetricArray(String proxyName, String metricName){
        String key = proxyName + "_" + metricName;
        List<Float> values = metrics.get(key);

        if (values == null || values.isEmpty()) {
            return 0;
        }

        return values.size();
    }

    public Float getLastMetricValue(String proxyName, String metricName) {
        String key = proxyName + "_" + metricName;
        List<Float> values = metrics.remove(key); // não remove do mapa

        if (values == null || values.isEmpty()) {
            return 0.0F;
        }

        // Retorna o último valor da lista
        return values.get(values.size() - 1);
    }


    public void reset(){
        metrics.clear();
    }

}