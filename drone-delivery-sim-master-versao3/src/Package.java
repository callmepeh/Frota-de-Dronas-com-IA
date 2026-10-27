

class Package {
    int id;
    double arrivalTime;
    double startServiceTime;
    double adExp; // Apenas para salvar no .csv o AD exponencial gerado para esse pacote

    Package(int id, double arrivalTime, double adExp) {
        this.id = id;
        this.arrivalTime = arrivalTime;
        this.startServiceTime = -1;
        this.adExp = adExp;
    }
}