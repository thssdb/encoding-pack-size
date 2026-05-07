package encoding.packsize;

import com.csvreader.CsvReader;
import com.csvreader.CsvWriter;
import org.junit.Assert;
import org.junit.Assume;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/** Lossless checks for decimal scaling used before bit-packing benchmarks. */
public class FloatToInteger {

  static final List<String> IGNORE_FILES =
      Arrays.asList(
          ".DS_Store",
          "full_data",
          "test.csv",
          "POI-lat.csv",
          "POI-lon.csv",
          "Basel-wind.csv",
          "Basel-temp.csv",
          "Air-sensor.csv",
          "Mem-usage.csv",
          "Cpu-usage_right.csv",
          "Disk-usage.csv",
          "init.csv");

  private static long[] scaleNumbers(List<String> numbers, int decimalMax) {
    BigDecimal scale = BigDecimal.TEN.pow(decimalMax);
    int size = numbers.size();
    long[] result = new long[size];

    if (size == 0) {
      return result;
    }

    BigDecimal min = null;
    BigDecimal[] scaledValues = new BigDecimal[size];

    for (int i = 0; i < size; i++) {
      BigDecimal val = new BigDecimal(numbers.get(i)).multiply(scale);
      scaledValues[i] = val;
      if (min == null || val.compareTo(min) < 0) {
        min = val;
      }
    }

    BigDecimal first = scaledValues[0].subtract(min);
    result[0] = first.toBigInteger().longValueExact();

    for (int i = 1; i < size; i++) {
      BigDecimal current = scaledValues[i].subtract(min);
      result[i] = current.toBigInteger().longValueExact();
    }

    return result;
  }

  @Test
  public void FloatToIntLosslessTest() throws IOException {
    String directory = "TestData";
    File dir = new File(directory);
    Assume.assumeTrue(
        "Skip: dataset directory missing: " + directory, dir.exists() && dir.isDirectory());

    final int batchSize = 1024;
    for (File file : Objects.requireNonNull(dir.listFiles())) {
      if (IGNORE_FILES.contains(file.getName()) || file.isDirectory()) {
        continue;
      }
      List<String> numbers = new ArrayList<>();
      List<Integer> decimalPlaces = new ArrayList<>();
      CsvReader csvReader = new CsvReader(file.getPath(), ',', StandardCharsets.UTF_8);
      try {
        while (csvReader.readRecord()) {
          for (String value : csvReader.getValues()) {
            String numStr = value.trim();
            if (!numStr.isEmpty()) {
              numbers.add(numStr);
              int decimal = 0;
              if (numStr.contains(".")) {
                String[] parts = numStr.split("\\.");
                decimal = parts[1].length();
              }
              decimalPlaces.add(decimal);
            }
          }
        }
      } finally {
        csvReader.close();
      }

      int decimalMax = decimalPlaces.stream().max(Integer::compare).orElse(0);
      BigDecimal scale = BigDecimal.TEN.pow(decimalMax);
      String dataset = file.getName();

      for (int i = 0; i < numbers.size(); i += batchSize) {
        int end = Math.min(numbers.size(), i + batchSize);
        List<String> batch = numbers.subList(i, end);
        int batchIndex = i / batchSize;
        long[] scaledBatch = scaleNumbers(batch, decimalMax);

        BigDecimal batchMin = null;
        BigDecimal[] scaledValues = new BigDecimal[batch.size()];
        for (int j = 0; j < batch.size(); j++) {
          BigDecimal val = new BigDecimal(batch.get(j)).multiply(scale);
          scaledValues[j] = val;
          if (batchMin == null || val.compareTo(batchMin) < 0) {
            batchMin = val;
          }
        }
        for (int j = 0; j < batch.size(); j++) {
          BigDecimal original = new BigDecimal(batch.get(j));
          BigDecimal scaledOnly = original.multiply(scale);
          BigDecimal restored = scaledOnly.divide(scale);
          Assert.assertEquals(
              dataset + " batch " + batchIndex + " cell " + j + ": x*10^d/10^d",
              0,
              original.compareTo(restored));

          BigInteger delta = scaledValues[j].subtract(batchMin).toBigInteger();
          Assert.assertTrue(
              dataset
                  + " batch "
                  + batchIndex
                  + " cell "
                  + j
                  + ": scaled delta must fit in long (bitLength="
                  + delta.bitLength()
                  + ")",
              delta.bitLength() <= 63);
          Assert.assertEquals(
              dataset + " batch " + batchIndex + " cell " + j + ": long delta vs scaleNumbers",
              delta.longValueExact(),
              scaledBatch[j]);
          BigDecimal decoded = BigDecimal.valueOf(scaledBatch[j]).add(batchMin).divide(scale);
          Assert.assertEquals(
              dataset + " batch " + batchIndex + " cell " + j + ": decode after min-offset",
              0,
              original.compareTo(decoded));
        }
      }
    }
  }

  /**
   * Export every numeric cell in {@code TestData/} as scaled longs (same batching and {@link
   * #scaleNumbers} as {@link #FloatToIntLosslessTest}), into {@code TestDataInt/}. For each input
   * file writes {@code <name>.csv} (one column {@code value}) and {@code <name>_blocks.csv} with
   * per-1024-block metadata: {@code decimal_max_used} is the file-wide max decimals used for
   * scaling; {@code decimal_max_in_block} is the max within that block only.
   */
  @Test
  public void exportTestDataIntFromFloatScaling() throws IOException {
    final String inDir = "TestData";
    final String outDir = "TestDataInt";
    final int batchSize = 1024;

    File dir = new File(inDir);
    Assume.assumeTrue(
        "Skip: dataset directory missing: " + inDir, dir.exists() && dir.isDirectory());

    File outRoot = new File(outDir);
    outRoot.mkdirs();

    for (File file : Objects.requireNonNull(dir.listFiles())) {
      if (IGNORE_FILES.contains(file.getName()) || file.isDirectory()) {
        continue;
      }
      List<String> numbers = new ArrayList<>();
      List<Integer> decimalPlaces = new ArrayList<>();
      CsvReader csvReader = new CsvReader(file.getPath(), ',', StandardCharsets.UTF_8);
      try {
        while (csvReader.readRecord()) {
          for (String value : csvReader.getValues()) {
            String numStr = value.trim();
            if (!numStr.isEmpty()) {
              numbers.add(numStr);
              int decimal = 0;
              if (numStr.contains(".")) {
                String[] parts = numStr.split("\\.");
                decimal = parts[1].length();
              }
              decimalPlaces.add(decimal);
            }
          }
        }
      } finally {
        csvReader.close();
      }

      int decimalMaxUsed = decimalPlaces.stream().max(Integer::compare).orElse(0);

      String baseName = file.getName();
      String intPath = outRoot.getPath() + "/" + baseName;
      String blockPath =
          outRoot.getPath()
              + "/"
              + (baseName.endsWith(".csv")
                  ? baseName.substring(0, baseName.length() - 4) + "_blocks.csv"
                  : baseName + "_blocks.csv");

      CsvWriter intWriter = new CsvWriter(intPath, ',', StandardCharsets.UTF_8);
      CsvWriter blockWriter = new CsvWriter(blockPath, ',', StandardCharsets.UTF_8);
      try {
        intWriter.writeRecord(new String[] {"value"});
        blockWriter.writeRecord(
            new String[] {
              "block_index", "block_size", "decimal_max_used", "decimal_max_in_block",
            });

        for (int i = 0; i < numbers.size(); i += batchSize) {
          int end = Math.min(numbers.size(), i + batchSize);
          List<String> batch = numbers.subList(i, end);
          List<Integer> batchDecimals = decimalPlaces.subList(i, end);
          int blockIndex = i / batchSize;
          int blockSize = end - i;

          int decimalMaxInBlock =
              batchDecimals.stream().max(Integer::compare).orElse(0);

          long[] scaled = scaleNumbers(batch, decimalMaxUsed);
          for (long v : scaled) {
            intWriter.writeRecord(new String[] {Long.toString(v)});
          }

          blockWriter.writeRecord(
              new String[] {
                Integer.toString(blockIndex),
                Integer.toString(blockSize),
                Integer.toString(decimalMaxUsed),
                Integer.toString(decimalMaxInBlock),
              });
        }
      } finally {
        intWriter.close();
        blockWriter.close();
      }

      System.out.println("Wrote " + intPath + " and " + blockPath);
    }
  }
}
