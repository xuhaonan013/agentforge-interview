package com.agentforge.modules.resume.model;

import com.agentforge.common.model.AsyncTaskStatus;
import com.agentforge.modules.interview.model.InterviewHistoryItemDTO;

import java.time.LocalDateTime;
import java.util.List;

/**
 * 简历详情DTO
 */
public record ResumeDetailDTO(
    Long id,
    String filename,
    Long fileSize,
    String contentType,
    String storageUrl,
    LocalDateTime uploadedAt,
    Integer accessCount,
    String resumeText,
    AsyncTaskStatus analyzeStatus,
    String analyzeError,
    List<AnalysisHistoryDTO> analyses,
    List<InterviewHistoryItemDTO> interviews
) {
    /**
     * 分析历史DTO
     */
    public record AnalysisHistoryDTO(
        Long id,
        Integer overallScore,
        Integer contentScore,
        Integer structureScore,
        Integer skillMatchScore,
        Integer expressionScore,
        Integer projectScore,
        String summary,
        LocalDateTime analyzedAt,
        List<String> strengths,
        List<Object> suggestions
    ) {}
}

