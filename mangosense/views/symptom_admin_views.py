from __future__ import annotations

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAdminUser
from rest_framework.response import Response

from mangosense.models import Disease, DiseaseSymptom, Symptom, SymptomAlias
from mangosense.serializers import (
    DiseaseSerializer,
    DiseaseSymptomSerializer,
    SymptomAliasSerializer,
    SymptomSerializer,
)

from mangosense.services.symptom_service import (
    service_create_alias,
    service_create_disease,
    service_create_disease_symptom,
    service_create_symptom,
    service_delete_alias,
    service_delete_disease,
    service_delete_disease_symptom,
    service_delete_symptom,
    service_update_alias,
    service_update_disease,
    service_update_disease_symptom,
    service_update_symptom,
)
#symptom
@api_view(['GET', 'POST'])
@permission_classes([IsAdminUser])
def symptom_list(request):
    if request.method == 'GET':
        plant_part = request.query_params.get('plant_part')
        queryString = Symptom.objects.all()
        if plant_part in ('leaf', 'fruit'):
            queryString = queryString.filter(plant_part=plant_part)
        return Response(SymptomSerializer(queryString, many=True).data)
    
    serializer = SymptomSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    symptom = service_create_symptom(serializer.validated_data)
    return Response(SymptomSerializer(symptom).data, status=status.HTTP_201_CREATED)

@api_view(['GET', 'PUT', 'DELETE'])
@permission_classes([IsAdminUser])
def symptom_detail(request, primaryKey):
    try:
        symptom = Symptom.objects.get(pk=primaryKey)
    except Symptom.DoesNotExist:
        return Response({'error': 'symptom not found'}, status=status.HTTP_404_NOT_FOUND)
    
    if request.method =='GET':
        return Response(SymptomSerializer(symptom).data)
    
    if request.method == 'PUT':
        serializer = SymptomSerializer(symptom, data=request.data, partial=True)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        updated = service_update_symptom(symptom, serializer.validated_data)
        return Response(SymptomSerializer(updated).data)
    
    #delete
    try:
        service_delete_symptom(symptom)
    except ValueError as exc:
        return Response({'error': str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    return Response(status=status.HTTP_204_NO_CONTENT)

#alieas list

@api_view(['GET', 'POST'])
@permission_classes([IsAdminUser])
def alias_list(request):
    if request.method == 'GET':
        return Response(
            SymptomAliasSerializer(SymptomAlias.objects.select_related('canonical').all(), many=True).data
        )
    
    serializer = SymptomAliasSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    alias = service_create_alias(serializer.validated_data)
    return Response(SymptomAliasSerializer(alias).data, status=status.HTTP_201_CREATED)


@api_view(['GET', 'PUT', 'DELETE'])
@permission_classes([IsAdminUser])
def alias_detail(request, primaryKey):
    try:
        alias = SymptomAlias.objects.select_related('canonical').get(pk=primaryKey)
    except SymptomAlias.DoesNotExist:
        return Response({'error': 'Alias not found'}, status=status.HTTP_404_NOT_FOUND)
    
    if request.method == 'GET':
        return Response(SymptomAliasSerializer(alias).data)
    
    if request.method == 'PUT':
        serializer = SymptomAliasSerializer(alias, data=request.data, partial=True)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        updated = service_update_alias(alias, serializer.validated_data)
        return Response(SymptomAliasSerializer(updated).data)
    
    service_delete_alias(alias)
    return Response(status=status.HTTP_204_NO_CONTENT)


#disease list

@api_view(['GET', 'POST'])
@permission_classes([IsAdminUser])
def disease_list(request):
    if request.method == 'GET':
        plant_part = request.query_params.get('plant_part')
        queryString = Disease.objects.all()
        if plant_part in ('leaf', 'fruit'):
            queryString = queryString.filter(plant_part=plant_part)
        return Response(DiseaseSerializer(queryString, many=True).data)

    serializer = DiseaseSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    disease = service_create_disease(serializer.validated_data)
    return Response(DiseaseSerializer(disease).data, status=status.HTTP_201_CREATED)


@api_view(['GET', 'PUT', 'DELETE'])
@permission_classes([IsAdminUser])
def disease_detail(request, primaryKey):
    try:
        disease = Disease.objects.get(pk=primaryKey)
    except Disease.DoesNotExist:
        return Response({'error': 'Disease not found'}, status=status.HTTP_404_NOT_FOUND)

    if request.method == 'GET':
        return Response(DiseaseSerializer(disease).data)

    if request.method == 'PUT':
        serializer = DiseaseSerializer(disease, data=request.data, partial=True)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        updated = service_update_disease(disease, serializer.validated_data)
        return Response(DiseaseSerializer(updated).data)

    service_delete_disease(disease)
    return Response(status=status.HTTP_204_NO_CONTENT)

#DiseaseSymptom 

@api_view(['GET', 'POST'])
@permission_classes([IsAdminUser])
def disease_symptom_list(request):
    if request.method == 'GET':
        disease_id = request.query_params.get('disease')
        qs = DiseaseSymptom.objects.select_related('disease', 'symptom').all()
        if disease_id:
            qs = qs.filter(disease_id=disease_id)
        return Response(DiseaseSymptomSerializer(qs, many=True).data)

    serializer = DiseaseSymptomSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    try:
        link = service_create_disease_symptom(serializer.validated_data)
    except ValueError as exc:
        return Response({'error': str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    return Response(DiseaseSymptomSerializer(link).data, status=status.HTTP_201_CREATED)



@api_view(['GET', 'PUT', 'DELETE'])
@permission_classes([IsAdminUser])
def disease_symptom_detail(request, primaryKey):
    try:
        link = DiseaseSymptom.objects.select_related('disease', 'symptom').get(pk=primaryKey)
    except DiseaseSymptom.DoesNotExist:
        return Response({'error': 'Link not found'}, status=status.HTTP_404_NOT_FOUND)

    if request.method == 'GET':
        return Response(DiseaseSymptomSerializer(link).data)

    if request.method == 'PUT':
        serializer = DiseaseSymptomSerializer(link, data=request.data, partial=True)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        try:
            updated = service_update_disease_symptom(link, serializer.validated_data)
        except ValueError as exc:
            return Response({'error': str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(DiseaseSymptomSerializer(updated).data)

    service_delete_disease_symptom(link)
    return Response(status=status.HTTP_204_NO_CONTENT)
    