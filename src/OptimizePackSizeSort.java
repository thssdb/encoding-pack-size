package encoding.packsize;

import com.csvreader.CsvReader;
import com.csvreader.CsvWriter;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** BP / BP+Prune+RMQ benchmarks on raw vs sorted chunks (Fig. 17 style); split from {@link OptimizePackSizeTest}. */
public class OptimizePackSizeSort extends OptimizePackSizeTest {

  @Test
  public void BPAll() throws IOException {
    System.out.println("\nPerformance Testing...");
    String directory = "TestData";
    String outputDirstr = OPTIMAL_PACK_RESULTS_BASE + "/output_BP_N2_all_no8";
    File outputDir = new File(outputDirstr);

    outputDir.mkdirs();
    File dir = new File(directory);
    for (File file : Objects.requireNonNull(dir.listFiles())) {

      if (IGNORE_FILES.contains(file.getName()) || file.isDirectory()) continue;
      System.out.println(file.getName());
      String Output = outputDirstr + "/" + file.getName();
      CsvWriter writer = new CsvWriter(Output, ',', StandardCharsets.UTF_8);

      String[] head = {
        "Input Direction",
        "Encoding Algorithm",
        "Encoding Time",
        "Decoding Time",
        "Points",
        "Compressed Size",
        "Compression Ratio"
      };
      writer.writeRecord(head);
      System.out.println("Processing " + file.getName() + "...");
      List<String> numbers = new ArrayList<>();
      List<Integer> decimalPlaces = new ArrayList<>();
      CsvReader csvReader = new CsvReader(file.getPath(), ',', StandardCharsets.UTF_8);
      while (csvReader.readRecord()) {
        for (String value : csvReader.getValues()) {
          String numStr = value.trim();
          if (!numStr.isEmpty()) {
            numbers.add(numStr);
            int decimal = 0, sigBits;
            if (numStr.contains(".")) {
              String[] parts = numStr.split("\\.");
              decimal = parts[1].length();
              sigBits = (int) ((parts[0].length() + decimal) * (Math.log(10) / Math.log(2)));
            } else {
              sigBits = (int) (numStr.length() * (Math.log(10) / Math.log(2)));
            }
            decimalPlaces.add(decimal);
          }
        }
      }
      int time_of_repeat = DEFAULT_BENCH_TIME_REPEAT;

      int decimalMax = decimalPlaces.stream().max(Integer::compare).orElse(0);

      int batchSize = 1024;
      List<long[]> batches = new ArrayList<>();

      for (int i = 0; i < numbers.size(); i += batchSize) {
        int end = Math.min(numbers.size(), i + batchSize);
        List<String> batch = numbers.subList(i, end);
        long[] scaledBatch = scaleNumbers(batch, decimalMax);
        batches.add(scaledBatch);
      }

      int totalLength = batches.stream().mapToInt(arr -> arr.length).sum();
      long[] scaledInts_all = new long[totalLength];

      int currentIndex = 0;
      for (long[] batch : batches) {
        System.arraycopy(batch, 0, scaledInts_all, currentIndex, batch.length);
        currentIndex += batch.length;
      }
      long[] costA = new long[1];
      long[] encA = new long[1];
      long[] decA = new long[1];
      benchChunkedBitPacking(
          scaledInts_all,
          numbers.size(),
          CHUNK_SIZE,
          time_of_repeat,
          chunk -> encodeChunkBitPacking(chunk, OptimizePackSizeallV2(chunk)),
          ec -> decodeBitPackingV2(ec.compressed, ec.bitWidths, ec.packSize, ec.nInts),
          costA,
          encA,
          decA);
      long modelCost = costA[0];
      long modelTime = encA[0];
      long modelDecodeTime = decA[0];

      double model_ratio = (double) modelCost / (double) (numbers.size() * 64);
      double modelTime_throughput = (double) (numbers.size() * 8000L) / (double) (modelTime);
      double modelDecodeTime_throughput =
          (double) (numbers.size() * 8000L) / (double) (modelDecodeTime);

      String[] record = {
        file.toString(),
        "BP+RMQ",
        String.valueOf(modelTime_throughput),
        String.valueOf(modelDecodeTime_throughput),
        String.valueOf(numbers.size()),
        String.valueOf(modelCost),
        String.valueOf(model_ratio)
      };
      writer.writeRecord(record);
      writer.close();

      System.out.println(
          "Optimal pack_size found, encoding throughput: " + modelTime_throughput + " MB/s");
      System.out.println("Decoding throughput: " + modelDecodeTime_throughput + " MB/s");
      System.out.println("Compression ratio: " + model_ratio);
    }
  }

  @Test
  public void BPAllSort() throws IOException {
    System.out.println("\nPerformance Testing...");
    String directory = "TestData";
    String outputDirstr = OPTIMAL_PACK_RESULTS_BASE + "/output_BP_N2_all_no8_sort";
    File outputDir = new File(outputDirstr);

    outputDir.mkdirs();
    File dir = new File(directory);
    for (File file : Objects.requireNonNull(dir.listFiles())) {

      if (IGNORE_FILES.contains(file.getName()) || file.isDirectory()) continue;
      if (!NO_TIME_SERIES_FILES.contains(file.getName())) continue;
      System.out.println(file.getName());
      String Output = outputDirstr + "/" + file.getName();
      CsvWriter writer = new CsvWriter(Output, ',', StandardCharsets.UTF_8);

      String[] head = {
        "Input Direction",
        "Encoding Algorithm",
        "Encoding Time",
        "Decoding Time",
        "Points",
        "Compressed Size",
        "Compression Ratio"
      };
      writer.writeRecord(head);
      System.out.println("Processing " + file.getName() + "...");
      List<String> numbers = new ArrayList<>();
      List<Integer> decimalPlaces = new ArrayList<>();
      CsvReader csvReader = new CsvReader(file.getPath(), ',', StandardCharsets.UTF_8);
      while (csvReader.readRecord()) {
        for (String value : csvReader.getValues()) {
          String numStr = value.trim();
          if (!numStr.isEmpty()) {
            numbers.add(numStr);
            int decimal = 0, sigBits;
            if (numStr.contains(".")) {
              String[] parts = numStr.split("\\.");
              decimal = parts[1].length();
              sigBits = (int) ((parts[0].length() + decimal) * (Math.log(10) / Math.log(2)));
            } else {
              sigBits = (int) (numStr.length() * (Math.log(10) / Math.log(2)));
            }
            decimalPlaces.add(decimal);
          }
        }
      }
      int time_of_repeat = DEFAULT_BENCH_TIME_REPEAT;

      int decimalMax = decimalPlaces.stream().max(Integer::compare).orElse(0);

      int batchSize = 1024;
      List<long[]> batches = new ArrayList<>();

      for (int i = 0; i < numbers.size(); i += batchSize) {
        int end = Math.min(numbers.size(), i + batchSize);
        List<String> batch = numbers.subList(i, end);
        long[] scaledBatch = scaleNumbers(batch, decimalMax);
        batches.add(scaledBatch);
      }

      int totalLength = batches.stream().mapToInt(arr -> arr.length).sum();
      long[] scaledInts_all = new long[totalLength];

      int currentIndex = 0;
      for (long[] batch : batches) {
        System.arraycopy(batch, 0, scaledInts_all, currentIndex, batch.length);
        currentIndex += batch.length;
      }
      long[] costA = new long[1];
      long[] encA = new long[1];
      long[] decA = new long[1];
      benchChunkedBitPacking(
          scaledInts_all,
          numbers.size(),
          CHUNK_SIZE,
          time_of_repeat,
          chunk -> {
            quickSortDesc(chunk, 0, chunk.length - 1);
            return encodeChunkBitPacking(chunk, OptimizePackSizeallForSort(chunk));
          },
          ec -> decodeBitPackingV2(ec.compressed, ec.bitWidths, ec.packSize, ec.nInts),
          costA,
          encA,
          decA);
      long modelCost = costA[0];
      long modelTime = encA[0];
      long modelDecodeTime = decA[0];

      double model_ratio = (double) modelCost / (double) (numbers.size() * 64);
      double modelTime_throughput = (double) (numbers.size() * 8000L) / (double) (modelTime);
      double modelDecodeTime_throughput =
          (double) (numbers.size() * 8000L) / (double) (modelDecodeTime);

      String[] record = {
        file.toString(),
        "BP+RMQ",
        String.valueOf(modelTime_throughput),
        String.valueOf(modelDecodeTime_throughput),
        String.valueOf(numbers.size()),
        String.valueOf(modelCost),
        String.valueOf(model_ratio)
      };
      writer.writeRecord(record);
      writer.close();

      System.out.println(
          "Optimal pack_size found, encoding throughput: " + modelTime_throughput + " MB/s");
      System.out.println("Decoding throughput: " + modelDecodeTime_throughput + " MB/s");
      System.out.println("Compression ratio: " + model_ratio);
    }
  }

  @Test
  public void BPPruneRMQ() throws IOException {
    System.out.println("\nPerformance Testing...");
    String directory = "TestData";
    String outputDirstr = OPTIMAL_PACK_RESULTS_BASE + "/output_BP_Prune_all_no8";
    File outputDir = new File(outputDirstr);

    outputDir.mkdirs();
    File dir = new File(directory);
    for (File file : Objects.requireNonNull(dir.listFiles())) {

      if (IGNORE_FILES.contains(file.getName()) || file.isDirectory()) continue;
      System.out.println(file.getName());
      String Output = outputDirstr + "/" + file.getName();
      CsvWriter writer = new CsvWriter(Output, ',', StandardCharsets.UTF_8);

      String[] head = {
        "Input Direction",
        "Encoding Algorithm",
        "Encoding Time",
        "Decoding Time",
        "Points",
        "Compressed Size",
        "Compression Ratio"
      };
      writer.writeRecord(head);
      System.out.println("Processing " + file.getName() + "...");
      List<String> numbers = new ArrayList<>();
      List<Integer> decimalPlaces = new ArrayList<>();
      CsvReader csvReader = new CsvReader(file.getPath(), ',', StandardCharsets.UTF_8);
      while (csvReader.readRecord()) {
        for (String value : csvReader.getValues()) {
          String numStr = value.trim();
          if (!numStr.isEmpty()) {
            numbers.add(numStr);
            int decimal = 0, sigBits;
            if (numStr.contains(".")) {
              String[] parts = numStr.split("\\.");
              decimal = parts[1].length();
              sigBits = (int) ((parts[0].length() + decimal) * (Math.log(10) / Math.log(2)));
            } else {
              sigBits = (int) (numStr.length() * (Math.log(10) / Math.log(2)));
            }
            decimalPlaces.add(decimal);
          }
        }
      }
      int time_of_repeat = DEFAULT_BENCH_TIME_REPEAT;

      int decimalMax = decimalPlaces.stream().max(Integer::compare).orElse(0);

      int batchSize = 1024;
      List<long[]> batches = new ArrayList<>();

      for (int i = 0; i < numbers.size(); i += batchSize) {
        int end = Math.min(numbers.size(), i + batchSize);
        List<String> batch = numbers.subList(i, end);
        long[] scaledBatch = scaleNumbers(batch, decimalMax);
        batches.add(scaledBatch);
      }

      int totalLength = batches.stream().mapToInt(arr -> arr.length).sum();
      long[] scaledInts_all = new long[totalLength];

      int currentIndex = 0;
      for (long[] batch : batches) {
        System.arraycopy(batch, 0, scaledInts_all, currentIndex, batch.length);
        currentIndex += batch.length;
      }
      long[] costA = new long[1];
      long[] encA = new long[1];
      long[] decA = new long[1];
      benchChunkedBitPacking(
          scaledInts_all,
          numbers.size(),
          CHUNK_SIZE,
          time_of_repeat,
          chunk -> encodeChunkBitPacking(chunk, DynamicPacking(chunk)),
          ec -> decodeBitPackingV2(ec.compressed, ec.bitWidths, ec.packSize, ec.nInts),
          costA,
          encA,
          decA);
      long modelCost = costA[0];
      long modelTime = encA[0];
      long modelDecodeTime = decA[0];

      double model_ratio = (double) modelCost / (double) (numbers.size() * 64);
      double modelTime_throughput = (double) (numbers.size() * 8000L) / (double) (modelTime);
      double modelDecodeTime_throughput =
          (double) (numbers.size() * 8000L) / (double) (modelDecodeTime);

      String[] record = {
        file.toString(),
        "BP+RMQ+Prune",
        String.valueOf(modelTime_throughput),
        String.valueOf(modelDecodeTime_throughput),
        String.valueOf(numbers.size()),
        String.valueOf(modelCost),
        String.valueOf(model_ratio)
      };
      writer.writeRecord(record);
      writer.close();

      System.out.println(
          "Optimal pack_size found, encoding throughput: " + modelTime_throughput + " MB/s");
      System.out.println("Decoding throughput: " + modelDecodeTime_throughput + " MB/s");
      System.out.println("Compression ratio: " + model_ratio);
    }
  }

  @Test
  public void BPPruneRMQSort() throws IOException {
    System.out.println("\nPerformance Testing...");
    String directory = "TestData";
    String outputDirstr = OPTIMAL_PACK_RESULTS_BASE + "/output_BP_RMQ_all_no8_sort";
    File outputDir = new File(outputDirstr);

    outputDir.mkdirs();
    File dir = new File(directory);
    for (File file : Objects.requireNonNull(dir.listFiles())) {

      if (IGNORE_FILES.contains(file.getName()) || file.isDirectory()) continue;
      if (!NO_TIME_SERIES_FILES.contains(file.getName())) continue;
      System.out.println(file.getName());
      String Output = outputDirstr + "/" + file.getName();
      CsvWriter writer = new CsvWriter(Output, ',', StandardCharsets.UTF_8);

      String[] head = {
        "Input Direction",
        "Encoding Algorithm",
        "Encoding Time",
        "Decoding Time",
        "Points",
        "Compressed Size",
        "Compression Ratio"
      };
      writer.writeRecord(head);
      System.out.println("Processing " + file.getName() + "...");
      List<String> numbers = new ArrayList<>();
      List<Integer> decimalPlaces = new ArrayList<>();
      CsvReader csvReader = new CsvReader(file.getPath(), ',', StandardCharsets.UTF_8);
      while (csvReader.readRecord()) {
        for (String value : csvReader.getValues()) {
          String numStr = value.trim();
          if (!numStr.isEmpty()) {
            numbers.add(numStr);
            int decimal = 0, sigBits;
            if (numStr.contains(".")) {
              String[] parts = numStr.split("\\.");
              decimal = parts[1].length();
              sigBits = (int) ((parts[0].length() + decimal) * (Math.log(10) / Math.log(2)));
            } else {
              sigBits = (int) (numStr.length() * (Math.log(10) / Math.log(2)));
            }
            decimalPlaces.add(decimal);
          }
        }
      }
      int time_of_repeat = DEFAULT_BENCH_TIME_REPEAT;

      int decimalMax = decimalPlaces.stream().max(Integer::compare).orElse(0);

      int batchSize = 1024;
      List<long[]> batches = new ArrayList<>();

      for (int i = 0; i < numbers.size(); i += batchSize) {
        int end = Math.min(numbers.size(), i + batchSize);
        List<String> batch = numbers.subList(i, end);
        long[] scaledBatch = scaleNumbers(batch, decimalMax);
        batches.add(scaledBatch);
      }

      int totalLength = batches.stream().mapToInt(arr -> arr.length).sum();
      long[] scaledInts_all = new long[totalLength];

      int currentIndex = 0;
      for (long[] batch : batches) {
        System.arraycopy(batch, 0, scaledInts_all, currentIndex, batch.length);
        currentIndex += batch.length;
      }
      long[] costA = new long[1];
      long[] encA = new long[1];
      long[] decA = new long[1];
      benchChunkedBitPacking(
          scaledInts_all,
          numbers.size(),
          CHUNK_SIZE,
          time_of_repeat,
          chunk -> {
            quickSortDesc(chunk, 0, chunk.length - 1);
            return encodeChunkBitPacking(chunk, OptimizePackSizeallV3ForSort(chunk));
          },
          ec -> decodeBitPackingV2(ec.compressed, ec.bitWidths, ec.packSize, ec.nInts),
          costA,
          encA,
          decA);
      long modelCost = costA[0];
      long modelTime = encA[0];
      long modelDecodeTime = decA[0];

      double model_ratio = (double) modelCost / (double) (numbers.size() * 64);
      double modelTime_throughput = (double) (numbers.size() * 8000L) / (double) (modelTime);
      double modelDecodeTime_throughput =
          (double) (numbers.size() * 8000L) / (double) (modelDecodeTime);

      String[] record = {
        file.toString(),
        "BP+RMQ",
        String.valueOf(modelTime_throughput),
        String.valueOf(modelDecodeTime_throughput),
        String.valueOf(numbers.size()),
        String.valueOf(modelCost),
        String.valueOf(model_ratio)
      };
      writer.writeRecord(record);
      writer.close();

      System.out.println(
          "Optimal pack_size found, encoding throughput: " + modelTime_throughput + " MB/s");
      System.out.println("Decoding throughput: " + modelDecodeTime_throughput + " MB/s");
      System.out.println("Compression ratio: " + model_ratio);
    }
  }
}
