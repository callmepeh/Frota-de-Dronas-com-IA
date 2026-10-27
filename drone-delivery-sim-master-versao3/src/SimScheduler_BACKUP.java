import org.ufpi.mercurycall.solvers.SPNsolver;
import org.ufpi.pasid.projects.monitor.ExecutionRequest;
import org.ufpi.pasid.projects.monitor.ExecutionResponse;
import org.ufpi.pasid.projects.monitor.ProducerWithReplyTo;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Agenda e executa eventos em ordem de tempo simulado, sincronizando com tempo real.
 */
class SimScheduler_BACKUP {

    private PriorityQueue<SimEvent> queueEvents = new PriorityQueue<>();
    private double currentTime = 0.0;
    private long realStartTime;      // tempo de início real da simulação
    private volatile boolean running = false;
    DatabaseManager monitorDT = DatabaseManager.getInstance();
    double simulationTime;
    SPNsolver spnSolver;

    //Map<String, Double> missionSteps = new LinkedHashMap<>();
    Map<String, String> dronesState = new HashMap<>();

    private ProducerWithReplyTo producer = new ProducerWithReplyTo();

    static class SystemState {
        double numDrones;
        double tamDeposito;
        double ad;
        double queue;
        double pousoDecola;
        double voo;

        // Novos valores
        double A;
        double B;
        double C;
        double D;
        double E;
    }


    /**
     * Agenda um evento para ser executado após um determinado atraso lógico (em ms).
     */
    void schedule(String proxyName, String name, double value, double delay, Runnable action) {
        queueEvents.add(new SimEvent(proxyName, name, value, currentTime + delay, action));
    }

    void resetSystemStateFile() {
        String filename = "system_state_log.csv";
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename, false))) {

            writer.write(
                    "Time(ms);" +
                            "AD;" +
                            "QUEUE;" +
                            "POUSO_DECOLA;" +
                            "VOO;" +
                            "MMT;" +
                            "throughput;" +
                            "A;" +
                            "B;" +
                            "C;" +
                            "D;" +
                            "E;" +
                            "numDrones\n"
            );

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private Map<String, Integer> countDroneStates(Map<String, String> dronesState) {
        Map<String, Integer> result = new HashMap<>();

        List<String> validStates = List.of(
                "POUSO_DECOLA1",
                "VOO1",
                "POUSO_DECOLA2",
                "POUSO_DECOLA3",
                "VOO2"
        );

        for (String state : dronesState.values()) {
            if (!validStates.contains(state)) continue;
            result.merge(state, 1, Integer::sum);
        }

        return result;
    }

    void saveSystemState(){
        String filename = "system_state_log.csv";

        Map<String, Integer> countedDroneStates = countDroneStates(dronesState);

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename, true))) {
            writer.write(String.format(
                    Locale.US,
                    "%.2f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%d;%d;%d;%d;%d;%d%n",
                    currentTime,
                    monitorDT.getAverageMetricValue("system", "AD"),
                    monitorDT.getAverageMetricValue("system", "QUEUE"),
                    monitorDT.getAverageMetricValue("system", "POUSO_DECOLA"),
                    monitorDT.getAverageMetricValue("system", "VOO"),
                    monitorDT.getAverageMetricValue("system", "MMT"),
                    monitorDT.getAverageMetricValue("system", "throughput"),

                    countedDroneStates.getOrDefault("POUSO_DECOLA1", 0),
                    countedDroneStates.getOrDefault("VOO1", 0),
                    countedDroneStates.getOrDefault("POUSO_DECOLA2", 0),
                    countedDroneStates.getOrDefault("POUSO_DECOLA3", 0),
                    countedDroneStates.getOrDefault("VOO2", 0),

                    Utils.numDronesAtivos
            ));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private List<ExecutionRequest> createScenario(SystemState s){
        List<ExecutionRequest> newBatch = new ArrayList<>();

        HashMap<String, Double> variables = new HashMap<>();
        double drones = Math.max(s.numDrones - (s.A + s.B + s.C + s.D + s.E), 0);
        double sumTokens = s.queue + (s.A + s.B + s.C + s.D + s.E);


        variables.put("numDrones", s.numDrones);
        variables.put("TOKENS", sumTokens);

        variables.put("AD", s.ad);

        variables.put("DEPOSITO", Math.abs(s.tamDeposito - Math.round(s.queue)));
        variables.put("PACOTES", (double) Math.round(s.queue));

        variables.put("DRONES", drones);

        variables.put("A", s.A);
        variables.put("B", s.B);
        variables.put("C", s.C);
        variables.put("D", s.D);
        variables.put("E", s.E);

        variables.put("VOO", s.voo);
        variables.put("POUSO_DECOL", s.pousoDecola);

        ExecutionRequest executionRequest = new ExecutionRequest();
        executionRequest.setConfigId("drones");
        //executionRequest.setModel(spnModelURL);
        executionRequest.setVariables(variables);
        executionRequest.setExecutionType(ExecutionRequest.ExecutionType.STATIONARY_SIMULATION);
        executionRequest.setModelType(ExecutionRequest.ModelType.SPN);
        executionRequest.setError(0.05);
        executionRequest.setMaxExecutionTimeInMinutes(0);
        executionRequest.setMetrics(Arrays.asList("MMT"));

        newBatch.add(executionRequest);

        return newBatch;
    }

    void runDigitalTwin(){
        // COLETA DOS DADOS
        Map<String, Integer> countedDroneStates = countDroneStates(dronesState);

        SystemState s = new SystemState();

        //s.numDrones = 10;
        s.tamDeposito = 1000;
        s.ad = monitorDT.getAverageMetricValue("system", "AD");
        s.queue = monitorDT.getAverageMetricValue("system", "QUEUE");
        s.pousoDecola = monitorDT.getAverageMetricValue("system", "POUSO_DECOLA");
        s.voo = monitorDT.getAverageMetricValue("system", "VOO");

        s.A = countedDroneStates.getOrDefault("POUSO_DECOLA1", 0);
        s.B = countedDroneStates.getOrDefault("VOO1", 0);
        s.C = countedDroneStates.getOrDefault("POUSO_DECOLA2", 0);
        s.D = countedDroneStates.getOrDefault("POUSO_DECOLA3", 0);
        s.E = countedDroneStates.getOrDefault("VOO2", 0);


        // SIMULAÇÃO DE CENÁRIOS
        int inicio = Utils.minDrones;
        int fim = Utils.maxDrones-1;
        int bestVarDrones = Utils.maxDrones;
        double limiteMMT = 5000000;

        while (inicio <= fim) {
            int meio = inicio + (fim - inicio) / 2;
            s.numDrones = meio;

            List<ExecutionRequest> batch = createScenario(s);

            // Envia e espera respostas
            Map<ExecutionRequest, ExecutionResponse> responses = producer.sendRequestsAndAwait(batch, "fila.fast");

            // Processa respostas
            for (ExecutionRequest req : batch) {
                ExecutionResponse resp = responses.get(req);
                Map<String, Double> metrics = resp.getMetricValues();

                double mmt = metrics.get("MMT");

                if (mmt < limiteMMT) {
                    if (s.numDrones < bestVarDrones) {
                        bestVarDrones = (int) s.numDrones;
                    }
                    fim = meio - 1;
                } else {
                    inicio = meio + 1;
                }
            }
        }

        // IMPLANTAÇÃO DA NOVA CONFIGURAÇÃO
        System.err.println("NEW CONFIG: " + bestVarDrones);
        Utils.numDronesAtivos = bestVarDrones;
        for (int i = 0; i < Utils.listaDeDrones.size(); i++) {
            Drone drone = Utils.listaDeDrones.get(i);

            if (i < Utils.numDronesAtivos) {
                // Ativa os drones que estão dentro do limite informado
                drone.isAvalaible = true;
            } else {
                // Desativa todos os drones que sobrarem (o resto)
                drone.isAvalaible = false;
            }
        }

    }



    /**
     * Executa a simulação sincronizada com o tempo real.
     */
    void run() {
        if (queueEvents.isEmpty()) return;

        realStartTime = System.currentTimeMillis();
        running = true;

        //missionSteps.put("POUSO_DECOLA1", 0.0);
        //missionSteps.put("POUSO_DECOLA2", 0.0);
        //missionSteps.put("POUSO_DECOLA3", 0.0);
        //missionSteps.put("POUSO_DECOLA4", 0.0);
        String spnModelURL = "https://raw.githubusercontent.com/lucassl2020/digital-twin-petri-nets/refs/heads/main/model_drones_validation_DT.xml";
        spnSolver = new SPNsolver(spnModelURL);

        resetSystemStateFile();
        while (running) {
            // Roda a cada x segundos
            if((currentTime % (60000*30)) < 1 && currentTime > 0){
                runDigitalTwin();
                saveSystemState();
            }

            if((currentTime % (60000)) < 1 && currentTime > 0){
                System.out.print("\rTempo: " + (currentTime/60000) + "minutos");

                monitorDT.addMetric(
                    "system",
                    "QUEUE",
                    (float) Utils.packageQueue.size(),
                    System.currentTimeMillis()
                );

                monitorDT.addMetric(
                    "system",
                    "throughput",
                    (float) Utils.tp,
                    System.currentTimeMillis()
                );

                Utils.tp = 0;
            }

            if(currentTime > simulationTime) {
                running = false;
            }else{
                while (queueEvents.peek().time <= currentTime) {
                    SimEvent event = queueEvents.poll();

                    if (!(event.name.contains("ToBase") || event.name.contains("END"))) {

                        if(event.name.contains("POUSO_DECOLA") || event.name.contains("VOO")) {
                            dronesState.put(event.proxyName, event.name);

                            if (event.name.contains("POUSO_DECOLA")) {
                                event.name = "POUSO_DECOLA";
                            } else if (event.name.contains("VOO")) {
                                event.name = "VOO";
                            }
                        }

                        monitorDT.addMetric(
                            "system",
                            event.name,
                            (float) event.value,
                            System.currentTimeMillis()
                        );

                    }

                    event.action.run();
                }

            }


            currentTime += 1;
        }
    }

    /**
     * Para a execução da simulação.
     */
    void stop() {
        running = false;
    }

    /**
     * Retorna o tempo simulado atual.
     */
    double now() {
        return currentTime;
    }
}
