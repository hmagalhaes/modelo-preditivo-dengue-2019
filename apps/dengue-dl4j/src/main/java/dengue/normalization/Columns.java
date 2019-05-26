package dengue.normalization;

public enum Columns {

	ANO(2000, 2030), MES(1, 12), MES_BOOLEAN(0, 1), TEMP_MAX_MEDIA(23, 43), TEMP_MEDIA(21, 39),
	UMIDADE_REL_MEDIA(22, 95), PRECIPITACAO_ACUMULADA(0, 650), VEL_MEDIA_VENTO(0, 4.1), NOTIFICACOES(0, 4000),
	ATINGIDOS(0, 2), POPULACAO(100_000, 1_000_000), INDICE(0, 3);

	public final double min;
	public final double max;

	Columns(double min, double max) {
		this.min = min;
		this.max = max;
	}

}
