package com.agentforge.modules.voiceinterview.service;

import com.agentforge.common.ai.LlmProviderRegistry;
import com.agentforge.modules.voiceinterview.config.VoiceInterviewProperties;
import com.agentforge.modules.voiceinterview.listener.VoiceEvaluateStreamProducer;
import com.agentforge.modules.voiceinterview.model.VoiceInterviewMessageEntity;
import com.agentforge.modules.voiceinterview.model.VoiceInterviewSessionEntity;
import com.agentforge.modules.voiceinterview.repository.VoiceInterviewEvaluationRepository;
import com.agentforge.modules.voiceinterview.repository.VoiceInterviewMessageRepository;
import com.agentforge.modules.voiceinterview.repository.VoiceInterviewSessionRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.redisson.api.RedissonClient;

import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
@DisplayName("语音面试摘要持久化")
class VoiceInterviewSummaryPersistenceTest {

  @Mock
  private VoiceInterviewSessionRepository sessionRepository;
  @Mock
  private VoiceInterviewMessageRepository messageRepository;
  @Mock
  private VoiceInterviewEvaluationRepository evaluationRepository;
  @Mock
  private RedissonClient redissonClient;
  @Mock
  private VoiceInterviewProperties properties;
  @Mock
  private VoiceEvaluateStreamProducer evaluateStreamProducer;
  @Mock
  private LlmProviderRegistry llmProviderRegistry;

  private VoiceInterviewService service;

  @BeforeEach
  void setUp() {
    service = new VoiceInterviewService(
        sessionRepository,
        messageRepository,
        evaluationRepository,
        redissonClient,
        properties,
        evaluateStreamProducer,
        llmProviderRegistry
    );
  }

  @Test
  @DisplayName("公开对话历史不应包含内部 SUMMARY 行")
  void conversationHistoryExcludesSummaryRows() {
    VoiceInterviewMessageEntity dialogue = VoiceInterviewMessageEntity.builder()
        .id(2L)
        .sessionId(42L)
        .messageType("DIALOGUE")
        .aiGeneratedText("请介绍一下项目")
        .sequenceNum(1)
        .build();
    when(messageRepository.findBySessionIdAndMessageTypeNotOrderBySequenceNumAsc(
        42L, VoiceInterviewMessageEntity.MESSAGE_TYPE_SUMMARY))
        .thenReturn(List.of(dialogue));

    List<VoiceInterviewMessageEntity> result = service.getConversationHistory("42");

    assertThat(result).containsExactly(dialogue);
  }

  @Test
  @DisplayName("更新摘要应复用已有行，避免先删后插的竞态窗口")
  void saveSummaryUpdatesExistingRow() {
    VoiceInterviewSessionEntity session = VoiceInterviewSessionEntity.builder().id(42L).build();
    VoiceInterviewMessageEntity existing = VoiceInterviewMessageEntity.builder()
        .id(7L)
        .sessionId(42L)
        .messageType(VoiceInterviewMessageEntity.MESSAGE_TYPE_SUMMARY)
        .aiGeneratedText("旧摘要")
        .sequenceNum(-6)
        .build();
    when(sessionRepository.findByIdForUpdate(42L)).thenReturn(Optional.of(session));
    when(messageRepository.findFirstBySessionIdAndMessageTypeOrderBySequenceNumAsc(
        42L, VoiceInterviewMessageEntity.MESSAGE_TYPE_SUMMARY))
        .thenReturn(Optional.of(existing));

    service.saveSummaryRow("42", "新摘要", 10);

    assertThat(existing.getAiGeneratedText()).isEqualTo("新摘要");
    assertThat(existing.getSequenceNum()).isEqualTo(-11);
    verify(messageRepository).save(existing);
    verify(messageRepository, never()).deleteBySessionId(42L);
  }
}
