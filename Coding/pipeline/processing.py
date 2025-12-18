"""
Data Processing Module
Handles data cleaning, transformation, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Tuple, Optional, List, Dict
import logging
import re

logger = logging.getLogger(__name__)


class DataProcessor:
    """Main data processing class"""

    def __init__(self):
        # Guardamos los scalers por método para poder reutilizarlos en test
        self.scalers: Dict[str, object] = {}

    def handle_missing_values(self, df: pd.DataFrame, method: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values

        IMPORTANTE:
        - Este método SOLO se debe usar DESPUÉS del split, sobre X_train / X_test.
        - No lo uses en clean_rows (antes del split), para evitar leakage.

        Args:
            df: Input DataFrame
            method: 'mean', 'median', 'drop', 'forward_fill'

        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()

        if method == 'drop':
            df = df.dropna()
        elif method in ['mean', 'median']:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    if method == 'mean':
                        df[col] = df[col].fillna(df[col].mean())
                    else:  # median
                        df[col] = df[col].fillna(df[col].median())
        elif method == 'forward_fill':
            df = df.fillna(method='ffill').fillna(method='bfill')

        return df

    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect and handle outliers

        NOTA: Este método NO elimina filas, solo recorta/sustituye valores.

        Args:
            df: Input DataFrame
            method: 'iqr', 'zscore'
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with outliers handled
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if method == 'iqr':
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df[col] = df[col].clip(lower_bound, upper_bound)

        elif method == 'zscore':
            from scipy import stats
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(df[col]))
                df[col] = df[col].mask(z_scores > threshold, df[col].mean())

        return df

    def scale_features(self, df: pd.DataFrame, method: str = 'standardscaler',
                       fit: bool = True) -> pd.DataFrame:
        """
        Scale numeric features

        Args:
            df: Input DataFrame
            method: 'standardscaler', 'minmaxscaler', 'robust'
            fit: Whether to fit the scaler (True en train, False en test)

        Returns:
            Scaled DataFrame
        """
        df = df.copy()
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)

        if len(numeric_cols) == 0:
            return df

        if method == 'standardscaler':
            scaler = StandardScaler()
        elif method == 'minmaxscaler':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaling method '{method}'. Skipping scaling.")
            return df

        if fit:
            # Guardamos el scaler para usarlo luego en el set de prueba
            self.scalers[method] = scaler
            scaled_data = scaler.fit_transform(df[numeric_cols])
            df[numeric_cols] = scaled_data
        else:
            if method in self.scalers:
                scaler = self.scalers[method]
                scaled_data = scaler.transform(df[numeric_cols])
                df[numeric_cols] = scaled_data
            else:
                logger.warning(
                    f"No trained scaler found for method '{method}'. "
                    f"Call scale_features(..., fit=True) on train before test."
                )

        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        return df.drop_duplicates().reset_index(drop=True)

    def remove_constant_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
        """
        Remove features with constant values

        Returns:
            DataFrame with constant features removed and list of removed features
        """
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        df = df.drop(columns=constant_cols)
        return df, constant_cols

    def correlate_features(self, df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """
        Remove highly correlated features

        Args:
            df: Input DataFrame
            threshold: Correlation threshold

        Returns:
            DataFrame with correlated features removed
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return df

        corr_matrix = df[numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        return df.drop(columns=to_drop)


    def normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names: lowercase and remove special characters

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with normalized column names
        """
        df = df.copy()
        
        def clean_name(col):
            # Convert to lowercase
            col = col.lower()
            # Remove special characters, keep only alphanumeric and underscore
            col = re.sub(r'[^a-z0-9_]', '', col)
            return col
        
        df.columns = [clean_name(col) for col in df.columns]
        logger.info(f"Column names normalized")
        return df

    def remove_categorical_columns(self, df: pd.DataFrame, categorical_list: List[str] = None) -> Tuple[pd.DataFrame, List]:
        """
        Remove specified categorical columns from DataFrame.
        Column names are normalized (lowercase, special chars removed) for matching.

        Args:
            df: Input DataFrame
            categorical_list: List of categorical column names to remove

        Returns:
            DataFrame with categorical columns removed and list of removed columns
        """
        df = df.copy()

        if categorical_list is None:
            categorical_list = [
                'codigo_almacen', 
                'almacen',
                'codigo_producto',
                'descripcion_producto',
                'cum',
                'tipo_producto',
                'descripcion_padre',
                'valor_promedio',
                'valor_final'
            ]

        # Normalize the list items
        categorical_list_normalized = [re.sub(r'[^a-z0-9_]', '', item.lower()) for item in categorical_list]

        # Find columns that match the categorical list
        cols_to_remove = []
        for col in df.columns:
            col_normalized = re.sub(r'[^a-z0-9_]', '', col.lower())
            if col_normalized in categorical_list_normalized:
                cols_to_remove.append(col)

        # Remove the columns
        if cols_to_remove:
            df = df.drop(columns=cols_to_remove)
            logger.info(f"Removed categorical columns: {cols_to_remove}")
        else:
            logger.info("No categorical columns found to remove")

        return df, cols_to_remove

    def select_usable_columns(self, df: pd.DataFrame, usable_columns: List[str] = None) -> Tuple[pd.DataFrame, List]:
        """
        Select only specified usable columns from DataFrame.
        Column names are normalized (lowercase, special chars removed) for matching.

        Args:
            df: Input DataFrame
            usable_columns: List of column names to keep

        Returns:
            DataFrame with only usable columns and list of selected columns
        """
        df = df.copy()

        if usable_columns is None:
            usable_columns = [
                '2024-1', '2024-2', '2024-3', '2024-4', '2024-5', '2024-6',
                '2024-7', '2024-8', '2024-9', '2024-10', '2024-11', '2024-12',
                '2025-1', '2025-2', '2025-3', '2025-4', '2025-5', '2025-6',
                '2025-7', '2025-8', '2025-9'
            ]

        # Normalize the usable columns list for matching
        usable_normalized = [re.sub(r'[^a-z0-9_]', '', col.lower()) for col in usable_columns]

        # Find columns that match the usable list
        cols_to_keep = []
        for col in df.columns:
            col_normalized = re.sub(r'[^a-z0-9_]', '', col.lower())
            if col_normalized in usable_normalized:
                cols_to_keep.append(col)

        # Select only the usable columns
        if cols_to_keep:
            df = df[cols_to_keep]
            logger.info(f"Selected usable columns: {cols_to_keep}")
        else:
            logger.info("No usable columns found")

        return df, cols_to_keep

    def filter_values(self, df: pd.DataFrame, remove_empty: bool = True,
                      remove_negative: bool = True,
                      value_range: Tuple[int, int] = (1, 100000)) -> Tuple[pd.DataFrame, dict]:
        """
        Filter DataFrame values based on specified criteria.
        IMPORTANT: Este método elimina filas. Usarlo SOLO en clean_rows (antes del split).

        Remove empty values, negative values, and keep only values within specified range.

        Args:
            df: Input DataFrame
            remove_empty: Whether to remove rows with empty/null values
            remove_negative: Whether to remove negative values
            value_range: Tuple of (min, max) for value filtering

        Returns:
            Filtered DataFrame and dictionary with filtering statistics
        """
        df = df.copy()
        initial_rows = len(df)
        filtering_stats = {}

        # Remove empty/null values
        if remove_empty:
            rows_before = len(df)
            df = df.dropna()
            rows_dropped = rows_before - len(df)
            filtering_stats['rows_dropped_empty'] = rows_dropped
            logger.info(f"Removed {rows_dropped} rows with empty values")

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Remove negative values and values outside range
        if remove_negative or value_range:
            for col in numeric_cols:
                rows_before = len(df)

                # Remove negative values
                if remove_negative:
                    df = df[df[col] >= 0]

                # Keep values within range
                if value_range:
                    min_val, max_val = value_range
                    # Mantener 0 o valores dentro del rango
                    df = df[(df[col] >= min_val) | (df[col] == 0)]
                    df = df[df[col] <= max_val]

                rows_dropped = rows_before - len(df)
                if rows_dropped > 0:
                    filtering_stats[f'rows_dropped_{col}'] = rows_dropped
                    logger.info(
                        f"Removed {rows_dropped} rows from column '{col}' "
                        f"due to value filtering"
                    )

        rows_final = len(df)
        total_dropped = initial_rows - rows_final
        filtering_stats['total_rows_initial'] = initial_rows
        filtering_stats['total_rows_final'] = rows_final
        filtering_stats['total_rows_dropped'] = total_dropped

        logger.info(
            f"Value filtering complete: {total_dropped} rows removed, {rows_final} rows remaining"
        )

        return df, filtering_stats

    # =====================================================================
    # NUEVA FASE: ANTES DEL SPLIT (clean_rows)
    # =====================================================================
    def clean_rows(self, df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, dict]:
        """
        Limpieza de filas ANTES del train/test split.
        Aquí SÍ se pueden eliminar filas sin romper la alineación X–y.

        Operaciones típicas:
        - Normalizar nombres de columnas
        - Eliminar duplicados
        - Filtrar filas con valores imposibles / NaN críticos
        - (Opcional) Filtrar negativos imposibles

        NO hace:
        - Imputación
        - Escalamiento
        - Transformaciones logarítmicas del target

        Args:
            df: DataFrame original completo (features + target)
            config: diccionario de configuración (ej. DATA_PARAMS)

        Returns:
            df_limpio, stats_dict
        """
        df = df.copy()
        stats = {}

        logger.info("Starting clean_rows (pre-split row cleaning)")

        # 1. Normalizar nombres de columnas
        df = self.normalize_column_names(df)

        # 2. Eliminar duplicados
        rows_before = len(df)
        df = self.remove_duplicates(df)
        stats['duplicates_removed'] = rows_before - len(df)
        logger.info(f"Removed {stats['duplicates_removed']} duplicate rows")

        # 3. Filtrar por valores imposibles / NaN críticos
        value_filter_config = config.get('value_filtering', {})
        if value_filter_config:
            df, filter_stats = self.filter_values(
                df,
                remove_empty=value_filter_config.get('remove_empty', False),
                remove_negative=value_filter_config.get('remove_negative', False),
                value_range=value_filter_config.get('value_range', (None, None)),
            )
            stats['value_filtering'] = filter_stats
        else:
            logger.info("Value filtering skipped in clean_rows (no config)")

        logger.info(f"Finished clean_rows. Rows remaining: {len(df)}")

        return df, stats

    # =====================================================================
    # FASE DESPUÉS DEL SPLIT (process_pipeline)
    # =====================================================================
    def process_pipeline(self, df: pd.DataFrame, config: dict, fit: bool = True, remove_categorical: bool = True,
                        categorical_list: List[str] = None, select_usable: bool = True, usable_columns: List[str] = None,
                        filter_values_flag: bool = False) -> pd.DataFrame:
        """
        Execute complete processing pipeline DESPUÉS del split.

        MUY IMPORTANTE:
        - Aquí NO se deben eliminar filas.
        - Solo transformaciones que mantengan el número de registros:
          * eliminar columnas categóricas
          * seleccionar columnas numéricas
          * imputación
          * outliers (por clipping, NO drop)
          * escalamiento
          * transformaciones matemáticas (log, sqrt, etc.)

        Args:
            df: Input DataFrame (X_train o X_test)
            config: Configuration dictionary with processing parameters
            fit: Whether to fit transformers (True en train, False en test)
            remove_categorical: Whether to remove categorical columns
            categorical_list: List of categorical columns to remove
            select_usable: Whether to select only usable columns
            usable_columns: List of usable columns to keep
            filter_values_flag: DEPRECATED aquí (filtrado debe ir en clean_rows)

        Returns:
            Processed DataFrame
        """
        logger.info("Starting data processing pipeline (post-split, no row drops)")

        df = df.copy()

        # Aviso si alguien intenta usar filter_values_flag aquí
        if filter_values_flag:
            logger.warning(
                "filter_values_flag=True was passed to process_pipeline, "
                "but row filtering must be done in clean_rows (pre-split). "
                "Filtering is ignored here to avoid misalignment between X and y."
            )

        # 1. Normalizar nombres de columnas (idempotente)
        df = self.normalize_column_names(df)

        # 2. Seleccionar columnas utilizables (solo columnas, no filas)
        if select_usable:
            df, selected = self.select_usable_columns(df, usable_columns)
            logger.info(f"Selected usable columns ({len(selected)} columns)")

        if df.shape[1] == 0:
            logger.error("No usable columns remaining after selection")
            raise ValueError("No usable columns remaining after selection")

        # 3. Eliminar columnas categóricas (afecta columnas, no filas)
        if remove_categorical:
            df, removed_cols = self.remove_categorical_columns(df, categorical_list)
            logger.info(f"Removed categorical columns: {removed_cols}")

        # 4. Manejo de NaN con imputación (NO drop)
        missing_method = config.get('handle_missing_values', 'mean')
        if missing_method == 'drop':
            logger.warning(
                "handle_missing_values='drop' is not allowed inside process_pipeline "
                "(no row removal after split). Using 'mean' instead."
            )
            missing_method = 'mean'

        df = self.handle_missing_values(df, method=missing_method)
        logger.info("Missing values handled by imputation")

        # 5. Manejo de outliers (solo modificaciones de valores, sin eliminar filas)
        outlier_method = config.get('outlier_method')
        if outlier_method:
            df = self.detect_outliers(df, method=outlier_method)
            logger.info("Outliers handled (value clipping)")
        else:
            logger.info("Outlier detection skipped (disabled in config)")

        # 6. Eliminar cualquier columna no numérica que haya sobrevivido
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            logger.info(f"Dropping remaining non-numeric columns: {list(non_numeric_cols)}")
            df = df[numeric_cols]

        # 7. Convertir negativos a 0 en las features (opcional pero recomendado)
        negative_count = (df < 0).sum().sum()
        if negative_count > 0:
            logger.info(f"Converting {negative_count} negative values to 0 in features")
            df = df.clip(lower=0)
        else:
            logger.info("No negative values found in features")

        # 8. Comprobar NaN restantes (si queda algo, imputar por media)
        if df.isnull().sum().sum() > 0:
            logger.warning(
                f"Remaining NaN values found: {df.isnull().sum().sum()}. "
                f"Filling with column mean."
            )
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    col_mean = df[col].mean()
                    if pd.isna(col_mean):
                        logger.warning(
                            f"Column {col} has all NaN values. Filling with 0."
                        )
                        df[col] = df[col].fillna(0)
                    else:
                        df[col] = df[col].fillna(col_mean)

        # 9. Escalar features (fit en train, transform en test)
        df = self.scale_features(df, method=config.get('feature_scaling', 'standardscaler'), fit=fit)
        logger.info("Features scaled")

        return df
