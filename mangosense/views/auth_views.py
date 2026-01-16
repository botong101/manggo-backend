from django.http import JsonResponse
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from django.shortcuts import render
from rest_framework_simplejwt.tokens import RefreshToken
from mangosense.models import UserProfile  # Import UserProfile model
from .utils import validate_password_strength  # Import from utils to avoid duplication
import json
import uuid

# validation stuff
def validate_name(name, field_name):
    """check name is valid"""
    errors = []
    if not name or len(name.strip()) < 2:
        errors.append(f"{field_name} must be at least 2 characters long.")
    return errors

def validate_address(address):
    """check address field"""
    errors = []
    if not address or len(address.strip()) < 5:
        errors.append("Address must be at least 5 characters long.")
    if len(address) > 200:
        errors.append("Address cannot exceed 200 characters.")
    return errors

# auth stuff
def register_view(request):
    if request.method == 'GET':
        return render(request, 'mangosense/register.html')

@csrf_exempt
@require_http_methods(["POST"])
def register_api(request):
    try:
        data = json.loads(request.body)
        first_name = (data.get('first_name') or data.get('firstName', '')).strip()
        last_name = (data.get('last_name') or data.get('lastName', '')).strip()
        
        # get location stuff from form
        province = data.get('province', '').strip()
        city = data.get('city', '').strip()
        barangay = data.get('barangay', '').strip()
        postal_code = data.get('postal_code') or data.get('postalCode', '').strip()
        
        # combine location into address or use whats there
        address = data.get('address', '').strip()
        if not address and province and city and barangay:
            address = f"{barangay}, {city}, {province} {postal_code}".strip()
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        confirm_password = data.get('confirm_password') or data.get('confirmPassword') or password

        errors = []

        # make sure we got everything - either address or location parts
        required_location = address or (province and city and barangay)
        if not all([first_name, last_name, required_location, email, password]):
            return JsonResponse({
                'success': False,
                'error': 'All required fields must be provided.',
                'debug': {
                    'firstName': bool(first_name),
                    'lastName': bool(last_name), 
                    'location': bool(required_location),
                    'email': bool(email),
                    'password': bool(password),
                    'received_fields': list(data.keys())
                }
            }, status=400)

        errors.extend(validate_name(first_name, "First name"))
        errors.extend(validate_name(last_name, "Last name"))
        
        # check address stuff
        if address:
            errors.extend(validate_address(address))
        elif province and city and barangay:
            # validate location parts
            if not province.strip():
                errors.append("Province is required.")
            if not city.strip():
                errors.append("City is required.")
            if not barangay.strip():
                errors.append("Barangay is required.")
        else:
            errors.append("Location information (address or province/city/barangay) is required.")

        try:
            validate_email(email)
        except ValidationError:
            errors.append("Invalid email format.")

        if User.objects.filter(email=email).exists():
            errors.append("An account with this email already exists.")

        if confirm_password and password != confirm_password:
            errors.append("Passwords do not match.")

        errors.extend(validate_password_strength(password))

        if errors:
            return JsonResponse({
                'success': False,
                'errors': errors
            }, status=400)

        # make unique username not email
        base_username = (first_name + last_name).lower() or "user"
        unique_username = f"{base_username}_{uuid.uuid4().hex[:8]}"

        user = User.objects.create_user(
            username=unique_username,  # use generated name
            first_name=first_name,
            last_name=last_name,
            email=email,
            password=password
        )

        # make profile with address
        UserProfile.objects.create(
            user=user,
            province=province,
            city=city,
            barangay=barangay,
            postal_code=postal_code,
            address=address
        )

        # make token for new user
        refresh = RefreshToken.for_user(user)
        access_token = refresh.access_token

        return JsonResponse({
            'success': True,
            'message': 'Account created successfully! You may now log in',
            'user_id': user.id,
            'access': str(access_token),
            'refresh': str(refresh),
            'user': {
                'id': user.id,
                'email': user.email,
                'firstName': user.first_name,
                'lastName': user.last_name,
                'full_name': f"{user.first_name} {user.last_name}"
            }
        })

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid data format.'
        }, status=400)
    except Exception as e:  
        return JsonResponse({
            'success': False,
            'error': f'An unexpected error occurred: {str(e)}'
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def login_api(request):
    try:
        data = json.loads(request.body)
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')

        if not email or not password:
            return JsonResponse({
                'success': False,
                'error': 'Email and password are required.'
            }, status=400)

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return JsonResponse({
                'success': False,
                'error': 'Invalid email or password.'
            }, status=404)

        user = authenticate(request, username=user.username, password=password)
        if user is not None and user.is_active:
            refresh = RefreshToken.for_user(user)
            access_token = refresh.access_token
            return JsonResponse({
                'success': True,
                'message': 'Login successful.',
                'access': str(access_token),
                'refresh': str(refresh),
                'user': {
                        'id': user.id,
                        'email': user.email,
                        'firstName': user.first_name,
                        'lastName': user.last_name,
                        'full_name': f"{user.first_name} {user.last_name}"
                    }
                })
        else:
            return JsonResponse({
                'success': False,
                'error': 'Invalid email or password.'
            }, status=401)
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid data format.'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'An unexpected error occurred: {str(e)}'
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def logout_api(request):
    if request.user.is_authenticated:
        logout(request)
        return JsonResponse({
            'success': True,
            'message': 'Logout successful.'
        })
    else:
        return JsonResponse({
            'success': False,
            'error': 'You are not logged in.'
        }, status=401)