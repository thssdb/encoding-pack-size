import com.github.Cwida.alp.ALPCompression;
import com.github.Cwida.alp.ALPDecompression;
import com.github.Tranway.buff.BuffCompressor;
import com.github.Tranway.buff.BuffDecompressor;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.CommonConfigurationKeys;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.io.compress.xerial.SnappyCodec;
import org.apache.hadoop.hbase.io.compress.xz.LzmaCodec;
import org.apache.hadoop.hbase.io.compress.zstd.ZstdCodec;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.compress.CompressionInputStream;
import org.apache.hadoop.io.compress.CompressionOutputStream;
import org.junit.jupiter.api.Test;
import org.urbcomp.startdb.selfstar.compressor.*;
import org.urbcomp.startdb.selfstar.compressor.xor.*;
import org.urbcomp.startdb.selfstar.decompressor.*;
import org.urbcomp.startdb.selfstar.decompressor.xor.*;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.*;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestCompressorPacksize {

    private static final String STORE_FILE = "src/test/resources/result/result.csv";
    private static final String STORE_PRUNING_FILE = "src/test/resources/result/resultPruningTime.csv";
    private static final String STORE_WINDOW_FILE = "src/test/resources/result/resultWindow.csv";
    private static final String STORE_BLOCK_FILE = "src/test/resources/result/resultBlock.csv";
    private static final double TIME_PRECISION = 1000.0;
    private static final int BLOCK_SIZE = 1000;
    private static final int NO_PARAM = 0;
    private static final String INIT_FILE = "init.csv";     // warm up memory and cpu
    private final String[] fileNames = {
            INIT_FILE,
            "Air-pressure.csv",
            "Air-sensor.csv",
            "Bird-migration.csv",
            "Bitcoin-price.csv",
            "Basel-temp.csv",
            "Basel-wind.csv",
            "Basel-temp.csv",
            "City-lat.csv",
            "City-lon.csv",
            "City-temp.csv",
            "Cpu-usage_right.csv",
            "Dew-point-temp.csv",
            "Disk-usage.csv",
            "electric_vehicle_charging.csv",
            "Food-price.csv",
            "SSD-bench.csv",
            "IR-bio-temp.csv",
            "PM10-dust.csv",
            "Stocks-DE.csv",
            "Stocks-UK.csv",
            "Stocks-USA.csv",
            "Wind-Speed.csv",
    };

    private final Map<String, Long> fileNameParamToTotalBits = new HashMap<>();
    private final Map<String, Long> fileNameParamToTotalBlock = new HashMap<>();
    private final Map<String, Long> fileNameParamMethodToCompressedBits = new HashMap<>();
    private final Map<String, Double> fileNameParamMethodToCompressTime = new HashMap<>();
    private final Map<String, Double> fileNameParamMethodToDecompressTime = new HashMap<>();
    private final Map<String, Double> fileNameParamMethodToCompressedRatio = new TreeMap<>();// use TreeMap to keep the order

    @Test
    public void testAllCompressor() {
        // 创建 per-dataset 输出目录
        File perDatasetDir = new File("/Users/xiaojinzhao/Documents/GitHub/encoding-pack-size/output_ElfStar");
        if (!perDatasetDir.exists()) {
            perDatasetDir.mkdirs();
        }

        for (String fileName : fileNames) {
            // 对于每个 dataset，先清理与该 dataset 相关的全局统计（防止残留）
            String fileNameParam = fileName + "," + NO_PARAM;
            // 初始化基础值（testALPCompressor/testFloatingCompressor 也会再初始化，但这里确保存在）
            fileNameParamToTotalBits.put(fileNameParam, 0L);
            fileNameParamToTotalBlock.put(fileNameParam, 0L);

            // 运行压缩/解压测试（ALP 和 Floating）
            testALPCompressor(fileName, NO_PARAM);
            testFloatingCompressor(fileName);

            // 如果是 INIT_FILE，跳过输出（与 BPTest 的行为一致）
            if (fileName.equals(INIT_FILE)) {
                continue;
            }

            // 收集并写入该数据集的 per-dataset CSV
            String prefix = fileName + "," + NO_PARAM + ","; // keys 格式为 fileName,param,method
            List<String> methodsForFile = new ArrayList<>();
            for (String key : fileNameParamMethodToCompressedBits.keySet()) {
                if (key.startsWith(prefix)) {
                    methodsForFile.add(key);
                }
            }

            // 如果没有任何方法数据，则跳过
            if (methodsForFile.isEmpty()) {
                System.out.println("No methods recorded for dataset " + fileName + " — skip per-dataset output.");
                continue;
            }

            // 获取点数与总比特数
            long totalBits = fileNameParamToTotalBits.getOrDefault(fileNameParam, 0L);
            long points = totalBits / 64L;

            File outFile = new File(perDatasetDir, fileName );
            try (FileWriter writer = new FileWriter(outFile, false)) {
                // header
                writer.write("Input Direction,Encoding Algorithm,Encoding Time,Decoding Time,Points,Compressed Size,Compression Ratio");
                writer.write("\n");

                System.out.println("Processing dataset: " + fileName);
                for (String key : methodsForFile) {
                    // key 格式为 fileName,param,method
                    String[] parts = key.split(",", 3);
                    String method = parts.length >= 3 ? parts[2] : "UNKNOWN";

                    Long compressedBits = fileNameParamMethodToCompressedBits.get(key);
                    Double compressTimeMicro = fileNameParamMethodToCompressTime.getOrDefault(key, 0.0);
                    Double decompressTimeMicro = fileNameParamMethodToDecompressTime.getOrDefault(key, 0.0);

                    if (compressedBits == null) {
                        // 跳过缺失数据的 method
                        continue;
                    }

                    // 计算吞吐 MB/s：MB = points*8 / (1024*1024); time_sec = time_micro / 1e6
                    double mb = (points * 8.0) / (1024.0 * 1024.0);
                    double encThroughput = 0.0;
                    double decThroughput = 0.0;
                    if (compressTimeMicro > 0.0) {
                        double timeSec = compressTimeMicro / 1_000_000.0;
                        encThroughput = timeSec > 0.0 ? (mb / timeSec) : 0.0;
                    }
                    if (decompressTimeMicro > 0.0) {
                        double timeSec = decompressTimeMicro / 1_000_000.0;
                        decThroughput = timeSec > 0.0 ? (mb / timeSec) : 0.0;
                    }

                    double ratio = totalBits > 0 ? (double) compressedBits / (double) totalBits : 0.0;

                    // 写 CSV 行
                    String record = String.join(",",
                            fileName,
                            method,
                            String.valueOf(encThroughput),
                            String.valueOf(decThroughput),
                            String.valueOf(points),
                            String.valueOf(compressedBits),
                            String.valueOf(ratio)
                    );
                    writer.write(record);
                    writer.write("\n");

                    // 控制台输出（类似 BPTest）
                    System.out.println("Method: " + method
                            + "  Encoding throughput: " + encThroughput + " MB/s"
                            + "  Decoding throughput: " + decThroughput + " MB/s"
                            + "  Compression ratio: " + ratio);
                }
                writer.flush();
                System.out.println("Per-dataset CSV written: " + outFile.getAbsolutePath());
            } catch (IOException e) {
                e.printStackTrace();
                System.err.println("Failed to write per-dataset CSV for " + fileName);
            }
        }

        // 在已有逻辑上继续计算全体的 ratio 并写入汇总文件（保留原有行为）
        fileNameParamMethodToCompressedBits.forEach((fileNameParamMethod, compressedBits) -> {
            String fileNameParam = fileNameParamMethod.split(",")[0] + "," + fileNameParamMethod.split(",")[1];
            long fileTotalBits = fileNameParamToTotalBits.get(fileNameParam);
            fileNameParamMethodToCompressedRatio.put(fileNameParamMethod, (compressedBits * 1.0) / fileTotalBits);
        });
        System.out.println("Test All Compressor - writing overall summary");
        writeResult(STORE_FILE, fileNameParamMethodToCompressedRatio, fileNameParamMethodToCompressTime, fileNameParamMethodToDecompressTime, fileNameParamToTotalBlock);
        System.gc();
    }

    private void testFloatingCompressor(String fileName) {
        String fileNameParam = fileName + "," + NO_PARAM;
        fileNameParamToTotalBits.put(fileNameParam, 0L);
        fileNameParamToTotalBlock.put(fileNameParam, 0L);
        ICompressor[] compressors = new ICompressor[]{
                new ElfStarCompressor(new ElfHuffXORCompressor()),
        };

        IDecompressor[] decompressors = new IDecompressor[]{
                new ElfStarDecompressor(new ElfHuffXORDecompressor()),
        };
        boolean firstMethod = true;
//        System.out.println(fileName);
        for (int i = 0; i < compressors.length; i++) {
            ICompressor compressor = compressors[i];
//            System.out.println(compressor.getKey());
            try (BlockReader br = new BlockReader(fileName, BLOCK_SIZE)) {
                List<Double> floatings;

                while ((floatings = br.nextBlock()) != null) {

                    double compressTime = 0;
                    double decompressTime;
                    if (floatings.size() != BLOCK_SIZE) {
                        break;
                    }
                    if (firstMethod) {
                        fileNameParamToTotalBits.put(fileNameParam, fileNameParamToTotalBits.get(fileNameParam) + floatings.size() * 64L);
                        fileNameParamToTotalBlock.put(fileNameParam, fileNameParamToTotalBlock.get(fileNameParam) + 1L);
                    }
                    double start = System.nanoTime();
                    floatings.forEach(compressor::addValue);
                    compressor.close();
                    compressTime += (System.nanoTime() - start) / TIME_PRECISION;
                    IDecompressor decompressor = decompressors[i];
                    decompressor.setBytes(compressor.getBytes());

                    start = System.nanoTime();
                    List<Double> deValues = decompressor.decompress();
                    decompressTime = (System.nanoTime() - start) / TIME_PRECISION;

                    assertEquals(deValues.size(), floatings.size());
                    for (int j = 0; j < floatings.size(); j++) {
                        assertEquals(floatings.get(j), deValues.get(j));
                    }
                    String fileNameParamMethod = fileName + "," + NO_PARAM + "," + compressor.getKey();
                    if (!fileNameParamMethodToCompressedBits.containsKey(fileNameParamMethod)) {
                        fileNameParamMethodToCompressedBits.put(fileNameParamMethod, compressor.getCompressedSizeInBits());
                        fileNameParamMethodToCompressTime.put(fileNameParamMethod, compressTime);
                        fileNameParamMethodToDecompressTime.put(fileNameParamMethod, decompressTime);
                    } else {
                        long newSize = fileNameParamMethodToCompressedBits.get(fileNameParamMethod) + compressor.getCompressedSizeInBits();
                        double newCTime = fileNameParamMethodToCompressTime.get(fileNameParamMethod) + compressTime;
                        double newDTime = fileNameParamMethodToDecompressTime.get(fileNameParamMethod) + decompressTime;
                        fileNameParamMethodToCompressedBits.put(fileNameParamMethod, newSize);
                        fileNameParamMethodToCompressTime.put(fileNameParamMethod, newCTime);
                        fileNameParamMethodToDecompressTime.put(fileNameParamMethod, newDTime);
                    }
                    compressor.refresh();
                    decompressor.refresh();
                }
            } catch (Exception e) {
                e.printStackTrace();
                throw new RuntimeException(fileName, e);
            }
            firstMethod = false;
        }
    }

    private void testALPCompressor(String fileName, int block) {
        long compressorBits;
        String fileNameParam = fileName + "," + block;
        if (block == NO_PARAM) {
            block = BLOCK_SIZE;
        }
        fileNameParamToTotalBits.put(fileNameParam, 0L);
        fileNameParamToTotalBlock.put(fileNameParam, 0L);
        double encodingDuration = 0;
        double decodingDuration = 0;
        try (BlockReader br = new BlockReader(fileName, block)) {
            List<List<List<Double>>> RowGroups = new ArrayList<>();
            List<List<Double>> floatingsList = new ArrayList<>();
            List<Double> floatings;
            int RGsize = 100;
            while ((floatings = br.nextBlock()) != null) {
                if (floatings.size() != block) {
                    break;
                }
                floatingsList.add(new ArrayList<>(floatings));
                fileNameParamToTotalBits.put(fileNameParam, fileNameParamToTotalBits.get(fileNameParam) + floatings.size() * 64L);
                if (floatingsList.size() == RGsize) {
                    RowGroups.add(new ArrayList<>(floatingsList));
                    floatingsList.clear();
                }
                fileNameParamToTotalBlock.put(fileNameParam, fileNameParamToTotalBlock.get(fileNameParam) + 1L);
            }
            if (!floatingsList.isEmpty()) {
                RowGroups.add(floatingsList);
            }

            long start = System.nanoTime();
            ALPCompression compressor = new ALPCompression(block);
            for (List<List<Double>> rowGroup : RowGroups) {
                compressor.entry(rowGroup);
                compressor.reset();
            }
            compressor.flush();
            encodingDuration += System.nanoTime() - start;

            byte[] result = compressor.getOut();

            start = System.nanoTime();
            ALPDecompression decompressor = new ALPDecompression(result);

            List<List<double[]>> deValues = new ArrayList<>();
            for (int i = 0; i < RowGroups.size(); i++) {
                List<double[]> deValue = decompressor.entry();
                deValues.add(deValue);
            }
            decodingDuration += System.nanoTime() - start;

            for (int RGidx = 0; RGidx < RowGroups.size(); RGidx++) {
                for (int i = 0; i < RowGroups.get(RGidx).size(); i++) {
                    for (int j = 0; j < RowGroups.get(RGidx).get(i).size(); j++) {
                        assertEquals(RowGroups.get(RGidx).get(i).get(j), deValues.get(RGidx).get(i)[j], "Value did not match");
                    }
                }
            }
            compressorBits = compressor.getSize();
            String fileNameParamMethod = fileNameParam + "," + "ALP";
            if (!fileNameParamMethodToCompressedBits.containsKey(fileNameParamMethod)) {
                fileNameParamMethodToCompressedBits.put(fileNameParamMethod, compressorBits);
                fileNameParamMethodToCompressTime.put(fileNameParamMethod, encodingDuration / TIME_PRECISION * BLOCK_SIZE / block);
                fileNameParamMethodToDecompressTime.put(fileNameParamMethod, decodingDuration / TIME_PRECISION * BLOCK_SIZE / block);
            } else {
                long newSize = fileNameParamMethodToCompressedBits.get(fileNameParamMethod) + compressorBits;
                double newCTime = fileNameParamMethodToCompressTime.get(fileNameParamMethod) + encodingDuration / TIME_PRECISION * BLOCK_SIZE / block;
                double newDTime = fileNameParamMethodToDecompressTime.get(fileNameParamMethod) + decodingDuration / TIME_PRECISION * BLOCK_SIZE / block;
                fileNameParamMethodToCompressedBits.put(fileNameParamMethod, newSize);
                fileNameParamMethodToCompressTime.put(fileNameParamMethod, newCTime);
                fileNameParamMethodToDecompressTime.put(fileNameParamMethod, newDTime);
            }

        } catch (Exception e) {
            throw new RuntimeException(fileName, e);
        }
    }

    private void writeResult(String storeFile,
                             Map<String, Double> fileNameParamMethodToRatio,
                             Map<String, Double> fileNameParamMethodToCTime,
                             Map<String, Double> fileNameParamMethodToDTime,
                             Map<String, Long> fileNameParamToTotalBlock) {
        Map<String, List<Double>> methodToRatios = new TreeMap<>();
        Map<String, List<Double>> methodToCTimes = new HashMap<>();
        Map<String, List<Double>> methodToDTimes = new HashMap<>();

        for (String fileNameParamMethod : fileNameParamMethodToRatio.keySet()) {
            String fileName = fileNameParamMethod.split(",")[0];
            String param = fileNameParamMethod.split(",")[1];
            String method = fileNameParamMethod.split(",")[2];
            String fileNameParam = fileName + "," + param;
            if (fileName.equals(INIT_FILE)) {
                continue;
            }
            String paramMethod = param + "\t" + method;
            if (!methodToRatios.containsKey(paramMethod)) {
                methodToRatios.put(paramMethod, new ArrayList<>());
                methodToCTimes.put(paramMethod, new ArrayList<>());
                methodToDTimes.put(paramMethod, new ArrayList<>());
            }
            methodToRatios.get(paramMethod).add(fileNameParamMethodToRatio.get(fileNameParamMethod));
            methodToCTimes.get(paramMethod).add(fileNameParamMethodToCTime.get(fileNameParamMethod) / fileNameParamToTotalBlock.get(fileNameParam));
            methodToDTimes.get(paramMethod).add(fileNameParamMethodToDTime.get(fileNameParamMethod) / fileNameParamToTotalBlock.get(fileNameParam));
        }

        System.out.println("Average Performance");
        System.out.println("Param\tMethod\tRatio\tCTime\tDTime");
        for (String paramMethod : methodToRatios.keySet()) {
            System.out.print(paramMethod + "\t");
            System.out.print(methodToRatios.get(paramMethod).stream().mapToDouble(o -> o).average().orElse(0) + "\t");
            System.out.print(methodToCTimes.get(paramMethod).stream().mapToDouble(o -> o).average().orElse(0) + "\t");
            System.out.println(methodToDTimes.get(paramMethod).stream().mapToDouble(o -> o).average().orElse(0));
        }

        try {
            File file = new File(storeFile).getParentFile();
            if (!file.exists() && !file.mkdirs()) {
                throw new IOException("Create directory failed: " + file);
            }
            try (FileWriter writer = new FileWriter(storeFile, true)) {
                writer.write("Dataset, Param, Method, Ratio, CTime, DTime");
                writer.write("\r\n");
                // 遍历键，并写入对应的值
                for (String fileNameParamMethod : fileNameParamMethodToRatio.keySet()) {
                    String fileNameParam = fileNameParamMethod.split(",")[0] + "," + fileNameParamMethod.split(",")[1];
                    writer.write(fileNameParamMethod);
                    writer.write(",");
                    writer.write(fileNameParamMethodToRatio.get(fileNameParamMethod).toString());
                    writer.write(",");
                    writer.write(fileNameParamMethodToCTime.get(fileNameParamMethod) / fileNameParamToTotalBlock.get(fileNameParam) + "");
                    writer.write(",");
                    writer.write(fileNameParamMethodToDTime.get(fileNameParamMethod) / fileNameParamToTotalBlock.get(fileNameParam) + "");
                    writer.write("\r\n");
                }
                System.out.println("Done!");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}