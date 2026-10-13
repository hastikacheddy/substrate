{{/* The image reference: a digest if one is given, else repository:tag. */}}
{{- define "substrate.image" -}}
{{- if .Values.image.digest -}}
{{ .Values.image.repository }}@{{ .Values.image.digest }}
{{- else -}}
{{ .Values.image.repository }}:{{ .Values.image.tag }}
{{- end -}}
{{- end -}}

{{/* Labels every object carries; the per-component name label is added where it is used. */}}
{{- define "substrate.partOf" -}}
app.kubernetes.io/part-of: substrate
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/instance: {{ .Release.Name }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{- end -}}

{{/* The restricted Pod Security Standard for a pod: non-root, a seccomp profile, a fixed user and group. */}}
{{- define "substrate.podSecurityContext" -}}
runAsNonRoot: true
runAsUser: {{ .user }}
runAsGroup: {{ .user }}
fsGroup: {{ .user }}
seccompProfile:
  type: RuntimeDefault
{{- end -}}

{{/* And for each container: no escalation, a read-only root, nothing added to the kernel's capabilities. */}}
{{- define "substrate.containerSecurityContext" -}}
allowPrivilegeEscalation: false
readOnlyRootFilesystem: true
capabilities:
  drop: [ALL]
{{- end -}}
